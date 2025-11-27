import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import random
import threading
import logging
import json
import os
from flask import Flask, render_template_string, jsonify
from concurrent.futures import ProcessPoolExecutor

# --- CONFIGURATION ---
DEVICE = torch.device("cpu")
CORE_COUNT = 4
TRAIN_FRAME_LIMIT = 5000   # Limit during training to speed up evolution
MIN_SHOW_SCORE = 100       # Minimum score to trigger a demo
GRID_W, GRID_H = 5, 10
SAVE_FILE = "neuro_save.json"

# --- SHARED STATE ---
SYSTEM_STATE = {
    'status': 'INITIALIZING HIVE...',
    'generation': 0,
    'best_score': 0.0,
    'best_genome': None, 
    'mode': 'TRAINING',
    'logs': [],
    'game_view': {},   
    'brain_view': {},
    'hyperparams': {} 
}

def add_log(msg):
    print(f"[SYS] {msg}")
    SYSTEM_STATE['logs'].insert(0, msg)
    if len(SYSTEM_STATE['logs']) > 20: SYSTEM_STATE['logs'].pop()

app = Flask(__name__)
log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)

# ==========================================
# 1. GENETICS & PHYSIOLOGY
# ==========================================

class GeneDecoder:
    @staticmethod
    def decode(vector):
        v = np.clip(vector, 0.0, 1.0)
        return {
            # Physiology
            'n_reservoir': int(v[0] * 80 + 20),     # Size: 20 to 100 Neurons
            'density': v[1] * 0.25 + 0.05,          # Sparsity: 5% to 30% Connected
            'leak_rate': v[2] * 0.9 + 0.01,         # Viscosity: 0.01 (Water) to 0.9 (Honey)
            'spectral_radius': v[3] * 1.2 + 0.5,    # Chaos: 0.5 (Stable) to 1.7 (Chaotic)
            # Learning
            'lr': 10 ** (-4.0 + (2.5 * v[4])),      # Learning Rate
            'input_gain': v[5] * 3.0 + 0.1          # Sensory Sensitivity
        }

class VisualCortex(nn.Module):
    """
    Pre-processes the grid. 
    The Reservoir doesn't see pixels; it sees 'features' extracted by this layer.
    """
    def __init__(self, input_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 16) # Compresses 50 pixels -> 16 signals
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x 

class DeepReservoir(nn.Module):
    """
    Echo State Network.
    Weights are FIXED (Non-trainable) and SPARSE.
    The "Intelligence" is in the dynamics (Echoes), not the weight adjustment.
    """
    def __init__(self, input_dim, params):
        super().__init__()
        self.size = params['n_reservoir']
        self.leak = params['leak_rate']
        
        # 1. Input Projection (Fixed)
        self.w_in = nn.Linear(input_dim, self.size, bias=False)
        with torch.no_grad():
            self.w_in.weight.uniform_(-params['input_gain'], params['input_gain'])
            self.w_in.weight.requires_grad_(False)

        # 2. Recurrent Weights (Fixed, Sparse)
        # Create random sparse mask
        mask = (torch.rand(self.size, self.size) < params['density']).float()
        # Create weights
        w_rec = (torch.rand(self.size, self.size) * 2 - 1) * mask
        # Enforce Spectral Radius (Critical for ESN stability)
        eigenvalues = torch.linalg.eigvals(w_rec)
        max_eig = torch.max(torch.abs(eigenvalues))
        if max_eig > 0:
            w_rec = w_rec * (params['spectral_radius'] / max_eig)
            
        self.w_rec = nn.Parameter(w_rec, requires_grad=False)
        
        # 3. Readout (TRAINABLE) - This is the only part that learns via Gradient Descent
        self.readout = nn.Linear(self.size, 3) 

        # Store adjacency list for the UI to draw lines
        indices = mask.nonzero().tolist()
        self.links = random.sample(indices, min(len(indices), 300)) # Limit UI links to 300 for performance

    def forward(self, u, h):
        # u: Input, h: Previous State
        # Update Equation: h(t) = (1-a)*h(t-1) + a*tanh(Win*u + Wrec*h(t-1))
        
        recurrence = F.linear(h, self.w_rec)
        injection = self.w_in(u)
        
        update = torch.tanh(injection + recurrence)
        h_new = (1 - self.leak) * h + self.leak * update
        
        # Readout
        logits = self.readout(h_new)
        return logits, h_new

class Agent(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # Input = Grid (50) + Player Pos (1) + Bias (1)
        self.vision = VisualCortex(GRID_W * GRID_H + 2) 
        self.brain = DeepReservoir(16, params) # 16 comes from vision output
        
    def forward(self, x, hidden):
        features = self.vision(x)
        return self.brain(features, hidden)

# ==========================================
# 2. ENVIRONMENT
# ==========================================

class GameEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.px = 2
        self.score = 0
        # Double Brick Logic
        self.bricks = [
            {'x': random.randint(0,4), 'y': 0},
            {'x': random.randint(0,4), 'y': -5}
        ]
        return self.get_obs()

    def get_obs(self):
        # Flatten grid representation
        grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        for b in self.bricks:
            if 0 <= b['y'] < GRID_H:
                grid[int(b['y']), int(b['x'])] = 1.0
        
        # Vector: [ ...50 pixel values..., normalized_player_x, bias_1.0 ]
        flat = torch.flatten(torch.tensor(grid))
        sensors = torch.tensor([self.px / (GRID_W-1), 1.0])
        return torch.cat([flat, sensors]).unsqueeze(0)

    def step(self, action):
        # Action: 0=Left, 1=Stay, 2=Right
        if action == 0: self.px = max(0, self.px - 1)
        elif action == 2: self.px = min(GRID_W - 1, self.px + 1)
        
        reward = 0.1 # Pulse of life reward
        done = False
        
        for b in self.bricks:
            b['y'] += 1
            
            # Hit Player?
            if b['y'] == GRID_H - 1 and b['x'] == self.px:
                reward = -5.0
                done = True
            
            # Hit Floor?
            if b['y'] >= GRID_H:
                b['y'] = -1
                b['x'] = random.randint(0, GRID_W - 1)
                self.score += 1
                reward = 1.0 # Point scored
                
        return self.get_obs(), reward, done

# ==========================================
# 3. SIMULATION WORKER
# ==========================================

def run_life_cycle(genome, generation):
    try:
        params = GeneDecoder.decode(genome)
        env = GameEnv()
        agent = Agent(params)
        
        # We only train the Readout and Vision layers.
        # The Reservoir structure is genetically fixed.
        optimizer = torch.optim.Adam([
            {'params': agent.vision.parameters()},
            {'params': agent.brain.readout.parameters()}
        ], lr=params['lr'])
        
        # --- LEARNING PHASE ---
        agent.train()
        # 60 episodes to learn how to use its body
        for _ in range(60):
            state = env.reset()
            h = torch.zeros(1, params['n_reservoir'])
            log_probs, rewards = [], []
            
            while True:
                logits, h = agent(state, h)
                dist = torch.distributions.Categorical(F.softmax(logits, dim=1))
                action = dist.sample()
                
                state, r, done = env.step(action.item())
                
                log_probs.append(dist.log_prob(action))
                rewards.append(r)
                if done: break
            
            # Reinforcement Learning Update (Policy Gradient)
            R = 0
            returns = []
            for r in reversed(rewards):
                R = r + 0.95 * R
                returns.insert(0, R)
            returns = torch.tensor(returns)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            loss = []
            for lp, ret in zip(log_probs, returns):
                loss.append(-lp * ret)
            
            if loss:
                optimizer.zero_grad()
                torch.stack(loss).sum().backward()
                optimizer.step()
        
        # --- TESTING PHASE ---
        agent.eval()
        state = env.reset()
        h = torch.zeros(1, params['n_reservoir'])
        fitness = 0
        
        while True:
            with torch.no_grad():
                logits, h = agent(state, h)
                action = torch.argmax(logits).item()
            
            state, _, done = env.step(action)
            fitness += 1
            
            # Cap training score to allow evolution to proceed
            if done or fitness > TRAIN_FRAME_LIMIT: break
            
        return {
            'fitness': fitness,
            'genome': genome,
            'params': params,
            'id': f"G{generation}-{random.randint(100,999)}"
        }
        
    except Exception as e:
        return {'fitness': 0, 'error': str(e)}

# ==========================================
# 4. CONTROLLER & PERSISTENCE
# ==========================================

def load_data():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, 'r') as f:
                data = json.load(f)
                SYSTEM_STATE['best_score'] = data.get('score', 0)
                SYSTEM_STATE['generation'] = data.get('gen', 0)
                SYSTEM_STATE['best_genome'] = data.get('genome', None)
                add_log(f"Loaded Save. Best Score: {SYSTEM_STATE['best_score']}")
                return
        except: pass
    add_log("No save found. Starting fresh.")

def save_data(score, gen, genome):
    with open(SAVE_FILE, 'w') as f:
        json.dump({
            'score': score,
            'gen': gen,
            'genome': genome.tolist() if isinstance(genome, np.ndarray) else genome
        }, f)

class EvolutionEngine(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        self.pop = [np.random.rand(6) for _ in range(CORE_COUNT * 3)]
        
    def run(self):
        load_data()
        gen = SYSTEM_STATE['generation'] + 1
        
        # Inject saved champion into population if exists
        if SYSTEM_STATE['best_genome'] is not None:
            self.pop[0] = np.array(SYSTEM_STATE['best_genome'])

        with ProcessPoolExecutor(max_workers=CORE_COUNT) as executor:
            while self.running:
                # If in demo mode, pause evolution loop
                while SYSTEM_STATE['mode'] == 'DEMO':
                    time.sleep(1)

                SYSTEM_STATE['status'] = f"EVOLVING BATCH {gen}"
                
                # Submit jobs
                futures = [executor.submit(run_life_cycle, ind, gen) for ind in self.pop]
                results = [f.result() for f in futures if 'error' not in f.result()]
                
                if not results: continue
                
                results.sort(key=lambda x: x['fitness'], reverse=True)
                best = results[0]
                
                # Update Records
                SYSTEM_STATE['generation'] = gen
                if best['fitness'] > SYSTEM_STATE['best_score']:
                    SYSTEM_STATE['best_score'] = best['fitness']
                    SYSTEM_STATE['best_genome'] = best['genome']
                    save_data(best['fitness'], gen, best['genome'])
                    add_log(f"🏆 NEW KING: {best['id']} ({best['fitness']:.0f} pts)")
                else:
                    add_log(f"Gen {gen} Best: {best['id']} ({best['fitness']:.0f} pts)")
                
                # Update Hyperparam Chart Data
                SYSTEM_STATE['hyperparams'] = best['params']
                
                # Decide to run Demo
                if best['fitness'] > MIN_SHOW_SCORE:
                    self.run_demo_mode(best)
                
                # Evolution (Elitism + Mutation)
                elites = [r['genome'] for r in results[:3]]
                next_pop = elites[:] # Keep elites
                
                # If we have a stored best that wasn't in this batch, add it back
                if SYSTEM_STATE['best_genome'] is not None:
                    next_pop.append(np.array(SYSTEM_STATE['best_genome']))
                
                while len(next_pop) < len(self.pop):
                    parent = random.choice(elites)
                    child = parent + np.random.normal(0, 0.06, 6) # Mutation
                    next_pop.append(np.clip(child, 0, 1))
                
                self.pop = next_pop
                gen += 1

    def run_demo_mode(self, agent_data):
        """
        Reconstructs the agent and lets it play UNTIL DEATH.
        Displays the visual cortex and reservoir activity.
        """
        SYSTEM_STATE['mode'] = 'DEMO'
        SYSTEM_STATE['current_id'] = agent_data['id']
        SYSTEM_STATE['status'] = "🔴 LIVE DEMO (NO LIMIT)"
        
        # Rebuild Agent
        params = agent_data['params']
        env = GameEnv()
        agent = Agent(params)
        
        # Quick Retrain (The 'Childhood' phase)
        optimizer = torch.optim.Adam([
            {'params': agent.vision.parameters()},
            {'params': agent.brain.readout.parameters()}
        ], lr=params['lr'])
        
        agent.train()
        # Train for 80 eps (invisible to user)
        for _ in range(80):
            s = env.reset(); h = torch.zeros(1, params['n_reservoir'])
            lp, rw = [], []
            while True:
                l, h = agent(s, h)
                dist = torch.distributions.Categorical(F.softmax(l, dim=1))
                a = dist.sample()
                s, r, d = env.step(a.item())
                lp.append(dist.log_prob(a)); rw.append(r)
                if d: break
            R=0; ret=[]
            for r in reversed(rw): R=r+0.95*R; ret.insert(0,R)
            ret = torch.tensor(ret)
            if len(ret)>1: ret = (ret - ret.mean())/(ret.std()+1e-9)
            loss = [(-l*r) for l,r in zip(lp, ret)]
            if loss: optimizer.zero_grad(); torch.stack(loss).sum().backward(); optimizer.step()

        # --- START LIVE SHOW ---
        agent.eval()
        state = env.reset()
        h = torch.zeros(1, params['n_reservoir'])
        
        while True:
            with torch.no_grad():
                # 1. Vision Features (Keep as Tensor for calc)
                vis_tensor = agent.vision(state)
                
                # 2. Forward Pass (Use Tensor)
                logits, h_new = agent.brain(vis_tensor, h)
                
                # 3. Convert to list ONLY for UI visualization
                vis_features = vis_tensor.tolist()[0]
                res_activations = h_new.tolist()[0]
                
                action = torch.argmax(logits).item()
            
            state, _, done = env.step(action)
            h = h_new
            
            # Send data to UI
            SYSTEM_STATE['game_view'] = {
                'px': env.px, 
                'bricks': env.bricks,
                'score': env.score
            }
            
            SYSTEM_STATE['brain_view'] = {
                'vis': vis_features,
                'res': res_activations,
                'links': agent.brain.links,
                'out': F.softmax(logits, dim=1).tolist()[0]
            }
            
            time.sleep(0.04) # ~25 FPS
            
            if done:
                add_log(f"Demo Ended. Score: {env.score}")
                time.sleep(1.5) # Let user see the death
                break
        
        SYSTEM_STATE['mode'] = 'TRAINING'

# ==========================================
# 5. UI
# ==========================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Neuro-Glass v4</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #050508; --panel: #0c0c10; --c1: #00ff9d; --c2: #00ccff; --err: #ff0055; }
        body { background: var(--bg); color: var(--c1); font-family: 'JetBrains Mono', monospace; margin: 0; height: 100vh; overflow: hidden; display: flex; }
        
        /* Sidebar */
        aside { width: 350px; background: var(--panel); border-right: 1px solid #222; padding: 20px; display: flex; flex-direction: column; gap: 15px; z-index: 10; }
        h1 { margin: 0; font-size: 28px; text-shadow: 0 0 15px rgba(0, 255, 157, 0.3); letter-spacing: -1px; }
        
        .card { background: #000; border: 1px solid #333; padding: 15px; border-radius: 6px; }
        .stat { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px; color: #888; }
        .val { color: #fff; font-weight: 800; }
        
        #logs { flex: 1; overflow-y: auto; font-size: 10px; color: #555; margin-top: 10px; border-top: 1px solid #222; padding-top: 10px; }
        .log-item { margin-bottom: 4px; }
        .log-item:first-child { color: #fff; }

        /* Main Stage */
        main { flex: 1; display: grid; grid-template-columns: 300px 1fr; grid-template-rows: 1fr; gap: 20px; padding: 30px; align-items: center; }
        
        .viz-box { background: #000; border: 1px solid #333; box-shadow: 0 0 40px rgba(0,0,0,0.5); border-radius: 8px; position: relative; overflow: hidden; height: 600px; }
        .viz-label { position: absolute; top: 15px; left: 15px; color: #444; font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; pointer-events: none; }
        
        canvas { display: block; width: 100%; height: 100%; }
        
        #status-bar { position: absolute; bottom: 20px; right: 30px; font-size: 12px; color: #444; }
        .demo-active { color: var(--c2) !important; text-shadow: 0 0 10px var(--c2); }
    </style>
</head>
<body>
    <aside>
        <h1>NEURO-GLASS</h1>
        <div style="font-size: 10px; color: #555; margin-bottom: 20px;">DEEP ECHO STATE NETWORK // EVOLUTION</div>
        
        <div class="card">
            <div class="stat"><span>STATUS</span><span id="status" class="val">INIT</span></div>
            <div class="stat"><span>GENERATION</span><span id="gen" class="val">0</span></div>
            <div class="stat"><span>RECORD</span><span id="best" class="val" style="color:var(--c2)">0</span></div>
            <div class="stat"><span>CURRENT ID</span><span id="aid" class="val">---</span></div>
        </div>

        <div class="card">
            <div style="color:#444; font-size:10px; margin-bottom:10px;">CURRENT ARCHITECTURE</div>
            <div class="stat"><span>NEURONS</span><span id="p-neu" class="val">0</span></div>
            <div class="stat"><span>LEAK RATE</span><span id="p-leak" class="val">0.0</span></div>
            <div class="stat"><span>SPARSITY</span><span id="p-den" class="val">0%</span></div>
            <div class="stat"><span>CHAOS</span><span id="p-chaos" class="val">0.0</span></div>
        </div>

        <div id="logs"></div>
    </aside>

    <main>
        <!-- Game -->
        <div class="viz-box">
            <div class="viz-label">Visual Input Stream</div>
            <canvas id="gCanvas" width="300" height="600"></canvas>
        </div>

        <!-- Brain -->
        <div class="viz-box" style="border-color: #222;">
            <div class="viz-label">Deep Reservoir State</div>
            <canvas id="bCanvas" width="800" height="600"></canvas>
        </div>
    </main>
    
    <div id="status-bar">SYSTEM READY</div>

    <script>
        const gc = document.getElementById('gCanvas').getContext('2d');
        const bc = document.getElementById('bCanvas').getContext('2d');
        
        // Neon Rect Helper
        function drawRect(ctx, x, y, w, h, c, glow=15) {
            ctx.shadowBlur = glow; ctx.shadowColor = c; ctx.fillStyle = c;
            ctx.fillRect(x, y, w, h); ctx.shadowBlur = 0;
        }

        // Generate organic positions for neurons based on ID hash simulation
        // We generate positions once and cache them so nodes don't jitter
        let nodeCache = [];
        function getNodes(count, w, h) {
            if (nodeCache.length !== count) {
                nodeCache = [];
                for(let i=0; i<count; i++) {
                    // Scatter randomly but keep away from edges
                    nodeCache.push({
                        x: 50 + Math.random() * (w - 100),
                        y: 50 + Math.random() * (h - 100)
                    });
                }
            }
            return nodeCache;
        }

        setInterval(() => {
            fetch('/status').then(r=>r.json()).then(d => {
                // Update Text
                document.getElementById('status').innerText = d.mode;
                document.getElementById('status').className = d.mode === 'DEMO' ? 'val demo-active' : 'val';
                document.getElementById('gen').innerText = d.gen;
                document.getElementById('best').innerText = d.score;
                document.getElementById('aid').innerText = d.id;
                document.getElementById('logs').innerHTML = d.logs.map(l=>`<div class="log-item">> ${l}</div>`).join('');

                // Params
                if(d.params) {
                    document.getElementById('p-neu').innerText = d.params.n_reservoir;
                    document.getElementById('p-leak').innerText = d.params.leak_rate.toFixed(2);
                    document.getElementById('p-den').innerText = (d.params.density*100).toFixed(0) + "%";
                    document.getElementById('p-chaos').innerText = d.params.spectral_radius.toFixed(2);
                }

                if (d.mode === 'DEMO' && d.game.px !== undefined) {
                    // --- DRAW GAME ---
                    gc.fillStyle = '#000'; gc.fillRect(0,0,300,600);
                    
                    // Grid Lines
                    gc.strokeStyle = '#111'; gc.beginPath();
                    for(let i=1; i<5; i++) { gc.moveTo(i*60, 0); gc.lineTo(i*60, 600); }
                    gc.stroke();

                    // Bricks
                    d.game.bricks.forEach(b => {
                        if(b.y >= 0) drawRect(gc, b.x*60+5, b.y*60+5, 50, 50, '#ff0055', 20);
                    });

                    // Player
                    drawRect(gc, d.game.px*60+5, 540+5, 50, 50, '#00ff9d', 20);
                    
                    // Score
                    gc.fillStyle = '#fff'; gc.font = '20px monospace'; gc.fillText(d.game.score, 20, 40);


                    // --- DRAW BRAIN ---
                    const bw = 800, bh = 600;
                    bc.fillStyle = 'rgba(0,0,0,0.25)'; bc.fillRect(0,0,bw,bh); // Trails

                    const brain = d.brain;
                    const nodes = getNodes(brain.res.length, bw, bh);

                    // 1. Draw Synapses (Links)
                    if (brain.links) {
                        bc.lineWidth = 1;
                        // Only draw links connected to active neurons to reduce clutter
                        brain.links.forEach(([src, dst]) => {
                            const val = Math.abs(brain.res[src]);
                            if (val > 0.1) {
                                const n1 = nodes[src];
                                const n2 = nodes[dst];
                                bc.beginPath(); bc.moveTo(n1.x, n1.y); bc.lineTo(n2.x, n2.y);
                                bc.strokeStyle = `rgba(0, 204, 255, ${val * 0.4})`;
                                bc.stroke();
                            }
                        });
                    }

                    // 2. Draw Neurons (Reservoir)
                    nodes.forEach((n, i) => {
                        const act = brain.res[i];
                        const size = 3 + Math.abs(act) * 10;
                        const color = act > 0 ? '#00ff9d' : '#ff0055';
                        
                        bc.beginPath(); bc.arc(n.x, n.y, size, 0, 6.28);
                        bc.fillStyle = color; 
                        bc.shadowBlur = size * 2; bc.shadowColor = color;
                        bc.fill(); bc.shadowBlur = 0;
                    });

                    // 3. Visual Cortex Inputs (Left Side)
                    brain.vis.forEach((v, i) => {
                        const y = 50 + i * 30;
                        const act = Math.max(0, v);
                        bc.fillStyle = `rgba(255, 255, 255, ${act})`;
                        bc.fillRect(10, y, 5, 5);
                        bc.fillText(`VIS_${i}`, 20, y+5);
                    });

                    // 4. Motor Outputs (Right Side)
                    brain.out.forEach((v, i) => {
                        const y = 200 + i * 60;
                        drawRect(bc, bw-40, y, 20, 20, `rgba(0, 255, 157, ${v})`, v*20);
                        bc.fillStyle = '#888'; bc.fillText(['LEFT','STAY','RIGHT'][i], bw-80, y+15);
                    });
                }
            });
        }, 50);
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/status')
def get_status():
    return jsonify({
        'status': SYSTEM_STATE['status'],
        'gen': SYSTEM_STATE['generation'],
        'score': SYSTEM_STATE['best_score'],
        'mode': SYSTEM_STATE['mode'],
        'id': SYSTEM_STATE['current_id'],
        'logs': SYSTEM_STATE['logs'],
        'game': SYSTEM_STATE['game_view'],
        'brain': SYSTEM_STATE['brain_view'],
        'params': SYSTEM_STATE.get('hyperparams', {})
    })

if __name__ == '__main__':
    engine = EvolutionEngine()
    engine.daemon = True
    engine.start()
    print("NEURO-GLASS v4 LIVE: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)