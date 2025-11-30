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
TRAIN_FRAME_LIMIT = 5000   # Limit during training to speed up throughput
MIN_SHOW_SCORE = 100       # Minimum score to trigger a demo automatically
GRID_W, GRID_H = 5, 10
SAVE_FILE = "neuro_control_save.json"

# FIXED HYPERPARAMETERS (The Control Variable)
FIXED_PARAMS = {
    'n_reservoir': 60,          # 60 Neurons (mid-range)
    'density': 0.15,            # 15% connectivity
    'leak_rate': 0.4,           # Medium viscosity
    'spectral_radius': 1.1,     # Slightly chaotic
    'lr': 0.001,                # Standard learning rate
    'input_gain': 1.0           # Moderate sensitivity
}

# --- SHARED STATE ---
SYSTEM_STATE = {
    'status': 'INITIALIZING SYSTEM...',
    'generation': 0,
    'best_score': 0.0,
    'best_weights': None,       # Stores the actual neural weights
    'mode': 'TRAINING',
    'logs': [],
    'game_view': {},   
    'brain_view': {},
    'hyperparams': FIXED_PARAMS,
    'runs_completed': 0,
    'current_id': '---',
    'manual_demo_request': False # Flag for button press
}

def add_log(msg):
    print(f"[SYS] {msg}")
    SYSTEM_STATE['logs'].insert(0, msg)
    if len(SYSTEM_STATE['logs']) > 20: SYSTEM_STATE['logs'].pop()

app = Flask(__name__)
log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)

# ==========================================
# 1. NETWORK ARCHITECTURE
# ==========================================

class VisualCortex(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 32)
        self.l2 = nn.Linear(32, 16) 
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = torch.tanh(self.l2(x))
        return x 

class DeepReservoir(nn.Module):
    def __init__(self, input_dim, params):
        super().__init__()
        self.size = params['n_reservoir']
        self.leak = params['leak_rate']
        
        self.w_in = nn.Linear(input_dim, self.size, bias=False)
        with torch.no_grad():
            self.w_in.weight.uniform_(-params['input_gain'], params['input_gain'])
            self.w_in.weight.requires_grad_(False)

        mask = (torch.rand(self.size, self.size) < params['density']).float()
        w_rec = (torch.rand(self.size, self.size) * 2 - 1) * mask
        eigenvalues = torch.linalg.eigvals(w_rec)
        max_eig = torch.max(torch.abs(eigenvalues))
        if max_eig > 0:
            w_rec = w_rec * (params['spectral_radius'] / max_eig)
            
        self.w_rec = nn.Parameter(w_rec, requires_grad=False)
        self.readout = nn.Linear(self.size, 3) 
        indices = mask.nonzero().tolist()
        self.links = random.sample(indices, min(len(indices), 300))

    def forward(self, u, h):
        recurrence = F.linear(h, self.w_rec)
        injection = self.w_in(u)
        update = torch.tanh(injection + recurrence)
        h_new = (1 - self.leak) * h + self.leak * update
        logits = self.readout(h_new)
        return logits, h_new

class Agent(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.vision = VisualCortex(GRID_W * GRID_H + 2) 
        self.brain = DeepReservoir(16, params)
        
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
        self.bricks = [
            {'x': random.randint(0,4), 'y': 0},
            {'x': random.randint(0,4), 'y': -5},
            {'x': random.randint(0,4), 'y': -10}
        ]
        return self.get_obs()

    def get_obs(self):
        grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        for b in self.bricks:
            if 0 <= b['y'] < GRID_H:
                grid[int(b['y']), int(b['x'])] = 1.0
        flat = torch.flatten(torch.tensor(grid))
        sensors = torch.tensor([self.px / (GRID_W-1), 1.0])
        return torch.cat([flat, sensors]).unsqueeze(0)

    def step(self, action):
        if action == 0: self.px = max(0, self.px - 1)
        elif action == 2: self.px = min(GRID_W - 1, self.px + 1)
        
        reward = 0.1 
        done = False
        
        for b in self.bricks:
            b['y'] += 1
            if b['y'] == GRID_H - 1 and b['x'] == self.px:
                reward = -5.0
                done = True
            if b['y'] >= GRID_H:
                b['y'] = -1
                b['x'] = random.randint(0, GRID_W - 1)
                self.score += 1
                reward = 1.0
                
        return self.get_obs(), reward, done

# ==========================================
# 3. SIMULATION WORKER
# ==========================================

def run_training_session(run_id):
    """
    Train an agent with FIXED hyperparameters.
    """
    try:
        params = FIXED_PARAMS
        env = GameEnv()
        agent = Agent(params)
        
        optimizer = torch.optim.Adam([
            {'params': agent.vision.parameters()},
            {'params': agent.brain.readout.parameters()}
        ], lr=params['lr'])
        
        # --- LEARNING PHASE ---
        agent.train()
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
            
            R = 0; returns = []
            for r in reversed(rewards): R = r + 0.95 * R; returns.insert(0, R)
            returns = torch.tensor(returns)
            if len(returns) > 1: returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            loss = [(-lp * ret) for lp, ret in zip(log_probs, returns)]
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
            if done: break
            # Hard cutoff for training speed
            if fitness > TRAIN_FRAME_LIMIT: break
        
        # EXTRACT WEIGHTS (Serialize for storage)
        weights = {k: v.cpu().numpy().tolist() for k, v in agent.state_dict().items()}
            
        return {
            'fitness': fitness,
            'params': params,
            'weights': weights,
            'id': f"R{run_id}-{random.randint(100,999)}"
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
                SYSTEM_STATE['runs_completed'] = data.get('runs', 0)
                SYSTEM_STATE['best_weights'] = data.get('weights', None)
                add_log(f"Loaded Save. Best Score: {SYSTEM_STATE['best_score']}")
                return
        except: pass
    add_log("No save found. Starting fresh.")

def save_data(score, runs, weights):
    with open(SAVE_FILE, 'w') as f:
        json.dump({
            'score': score,
            'runs': runs,
            'weights': weights
        }, f)

class TrainingEngine(threading.Thread):
    def __init__(self):
        super().__init__()
        self.running = True
        
    def run(self):
        global MIN_SHOW_SCORE
        
        load_data()
        run_count = SYSTEM_STATE['runs_completed'] + 1

        # On Startup: If we have a champion, show it immediately
        if SYSTEM_STATE['best_weights'] is not None:
            add_log("🌟 WAKING CHAMPION...")
            self.trigger_manual_demo()

        with ProcessPoolExecutor(max_workers=CORE_COUNT) as executor:
            while self.running:
                # Pause for Demo
                while SYSTEM_STATE['mode'] == 'DEMO':
                    time.sleep(1)

                # Check Manual Trigger
                if SYSTEM_STATE['manual_demo_request']:
                    SYSTEM_STATE['manual_demo_request'] = False
                    if SYSTEM_STATE['best_weights'] is not None:
                        add_log("👁️ DEMO REQUESTED")
                        self.trigger_manual_demo()
                        continue
                    else:
                        add_log("Cannot demo: No champion yet.")

                SYSTEM_STATE['status'] = f"TRAINING BATCH {run_count}"
                
                # Submit parallel training jobs
                futures = [executor.submit(run_training_session, run_count) for _ in range(CORE_COUNT * 3)]
                results = [f.result() for f in futures if 'error' not in f.result()]
                
                if not results: continue
                
                results.sort(key=lambda x: x['fitness'], reverse=True)
                best = results[0]
                
                # Update Records
                SYSTEM_STATE['runs_completed'] = run_count
                
                # Decision Logic
                should_demo = False
                if best['fitness'] >= MIN_SHOW_SCORE: should_demo = True
                if best['fitness'] > SYSTEM_STATE['best_score'] and best['fitness'] > 100: should_demo = True
                if best['fitness'] > TRAIN_FRAME_LIMIT:
                    should_demo = True
                    add_log("🚀 LIMIT BROKEN! STARTING UNLIMITED DEMO...")

                # New Record?
                if best['fitness'] > SYSTEM_STATE['best_score']:
                    SYSTEM_STATE['best_score'] = best['fitness']
                    SYSTEM_STATE['best_weights'] = best['weights']
                    
                    save_data(best['fitness'], run_count, best['weights'])
                    
                    add_log(f"🏆 NEW RECORD: {best['id']} ({best['fitness']:.0f} pts)")
                    if best['fitness'] > 5000:
                        MIN_SHOW_SCORE = best['fitness'] + 100
                        add_log(f"📈 Threshold raised to {MIN_SHOW_SCORE}")
                else:
                    add_log(f"Batch {run_count} Best: {best['id']} ({best['fitness']:.0f} pts)")
                
                # Auto Demo
                if should_demo:
                    self.run_demo_mode(best)
                
                run_count += 1

    def trigger_manual_demo(self):
        # Construct a fake agent data packet using the stored best state
        champion_data = {
            'fitness': SYSTEM_STATE['best_score'],
            'params': FIXED_PARAMS,
            'id': "CHAMPION",
            'weights': SYSTEM_STATE['best_weights']
        }
        self.run_demo_mode(champion_data)

    def run_demo_mode(self, agent_data):
        """
        Reconstructs the agent using saved weights.
        """
        SYSTEM_STATE['mode'] = 'DEMO'
        SYSTEM_STATE['current_id'] = agent_data['id']
        SYSTEM_STATE['status'] = "🔴 LIVE DEMO (NO LIMIT)"
        
        params = agent_data['params']
        env = GameEnv()
        agent = Agent(params)
        
        # Load Weights if available
        if 'weights' in agent_data and agent_data['weights'] is not None:
            saved_state = {k: torch.tensor(v) for k, v in agent_data['weights'].items()}
            agent.load_state_dict(saved_state)
            add_log("🧠 Brain weights loaded.")
        else:
            # Fallback for old saves
            add_log("⚠️ Retraining clone (No weights found)...")
            optimizer = torch.optim.Adam([
                {'params': agent.vision.parameters()},
                {'params': agent.brain.readout.parameters()}
            ], lr=params['lr'])
            
            agent.train()
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
                vis_tensor = agent.vision(state)
                logits, h_new = agent.brain(vis_tensor, h)
                vis_features = vis_tensor.tolist()[0]
                res_activations = h_new.tolist()[0]
                action = torch.argmax(logits).item()
            
            state, _, done = env.step(action)
            h = h_new
            
            SYSTEM_STATE['game_view'] = {
                'px': env.px, 'bricks': env.bricks, 'score': env.score
            }
            SYSTEM_STATE['brain_view'] = {
                'vis': vis_features,
                'res': res_activations,
                'links': agent.brain.links,
                'out': F.softmax(logits, dim=1).tolist()[0]
            }
            
            time.sleep(0.04) 
            
            if done:
                add_log(f"💀 Demo Ended. Final Score: {env.score}")
                time.sleep(1.5)
                break
        
        SYSTEM_STATE['mode'] = 'TRAINING'

# ==========================================
# 5. UI
# ==========================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Neuro-Glass CONTROL</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;800&display=swap" rel="stylesheet">
    <style>
        :root { --bg: #050508; --panel: #0c0c10; --c1: #ffaa00; --c2: #ff6600; --err: #ff0055; }
        body { background: var(--bg); color: var(--c1); font-family: 'JetBrains Mono', monospace; margin: 0; height: 100vh; overflow: hidden; display: flex; }
        aside { width: 350px; background: var(--panel); border-right: 1px solid #222; padding: 20px; display: flex; flex-direction: column; gap: 15px; z-index: 10; }
        h1 { margin: 0; font-size: 28px; text-shadow: 0 0 15px rgba(255, 170, 0, 0.3); letter-spacing: -1px; }
        
        .card { background: #000; border: 1px solid #333; padding: 15px; border-radius: 6px; }
        .stat { display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 12px; color: #888; }
        .val { color: #fff; font-weight: 800; }
        
        #logs { flex: 1; overflow-y: auto; font-size: 10px; color: #555; margin-top: 10px; border-top: 1px solid #222; padding-top: 10px; }
        .log-item { margin-bottom: 4px; }
        .log-item:first-child { color: #fff; }

        button { 
            background: #222; border: 1px solid #444; color: #fff; 
            padding: 15px; cursor: pointer; font-family: inherit; font-weight: bold; 
            font-size: 14px; width: 100%; transition: 0.2s; 
            text-transform: uppercase; letter-spacing: 1px;
            box-shadow: 0 4px 0 #111;
        }
        button:hover { background: var(--c2); color: #000; border-color: var(--c2); box-shadow: 0 4px 0 #aa4400; }
        button:active { transform: translateY(4px); box-shadow: none; }

        .control-badge { background: var(--c2); color: #000; padding: 3px 8px; border-radius: 3px; font-size: 9px; font-weight: 800; }

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
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h1>NEURO-GLASS</h1>
            <span class="control-badge">CONTROL</span>
        </div>
        <div style="font-size: 10px; color: #555; margin-bottom: 20px;">FIXED HYPERPARAMETERS // NO EVOLUTION</div>
        
        <div class="card">
            <div class="stat"><span>STATUS</span><span id="status" class="val">INIT</span></div>
            <div class="stat"><span>RUNS COMPLETED</span><span id="gen" class="val">0</span></div>
            <div class="stat"><span>RECORD</span><span id="best" class="val" style="color:var(--c2)">0</span></div>
            <div class="stat"><span>CURRENT ID</span><span id="aid" class="val">---</span></div>
        </div>

        <button onclick="triggerDemo()">DEMO BEST</button>

        <div class="card">
            <div style="color:#444; font-size:10px; margin-bottom:10px;">FIXED ARCHITECTURE</div>
            <div class="stat"><span>NEURONS</span><span id="p-neu" class="val">60</span></div>
            <div class="stat"><span>LEAK RATE</span><span id="p-leak" class="val">0.40</span></div>
            <div class="stat"><span>SPARSITY</span><span id="p-den" class="val">15%</span></div>
            <div class="stat"><span>CHAOS</span><span id="p-chaos" class="val">1.10</span></div>
            <div class="stat"><span>LEARNING RATE</span><span id="p-lr" class="val">0.001</span></div>
        </div>

        <div id="logs"></div>
    </aside>

    <main>
        <div class="viz-box">
            <div class="viz-label">Visual Input Stream</div>
            <canvas id="gCanvas" width="300" height="600"></canvas>
        </div>
        <div class="viz-box" style="border-color: #222;">
            <div class="viz-label">Deep Reservoir State</div>
            <canvas id="bCanvas" width="800" height="600"></canvas>
        </div>
    </main>
    
    <div id="status-bar">SYSTEM READY</div>

    <script>
        const gc = document.getElementById('gCanvas').getContext('2d');
        const bc = document.getElementById('bCanvas').getContext('2d');
        
        function triggerDemo() {
            fetch('/demo_best').then(r=>r.json()).then(d => {
                if(d.status !== 'ok') alert("No champion saved yet to demo!");
            });
        }

        function drawRect(ctx, x, y, w, h, c, glow=15) {
            ctx.shadowBlur = glow; ctx.shadowColor = c; ctx.fillStyle = c;
            ctx.fillRect(x, y, w, h); ctx.shadowBlur = 0;
        }

        let nodeCache = [];
        function getNodes(count, w, h) {
            if (nodeCache.length !== count) {
                nodeCache = [];
                for(let i=0; i<count; i++) {
                    nodeCache.push({ x: 50 + Math.random() * (w - 100), y: 50 + Math.random() * (h - 100) });
                }
            }
            return nodeCache;
        }

        setInterval(() => {
            fetch('/status').then(r=>r.json()).then(d => {
                document.getElementById('status').innerText = d.mode;
                document.getElementById('status').className = d.mode === 'DEMO' ? 'val demo-active' : 'val';
                document.getElementById('gen').innerText = d.runs;
                document.getElementById('best').innerText = d.score;
                document.getElementById('aid').innerText = d.id;
                document.getElementById('logs').innerHTML = d.logs.map(l=>`<div class="log-item">> ${l}</div>`).join('');

                if (d.mode === 'DEMO' && d.game.px !== undefined) {
                    gc.fillStyle = '#000'; gc.fillRect(0,0,300,600);
                    gc.strokeStyle = '#111'; gc.beginPath();
                    for(let i=1; i<5; i++) { gc.moveTo(i*60, 0); gc.lineTo(i*60, 600); }
                    gc.stroke();

                    d.game.bricks.forEach(b => {
                        if(b.y >= 0) drawRect(gc, b.x*60+5, b.y*60+5, 50, 50, '#ff0055', 20);
                    });
                    drawRect(gc, d.game.px*60+5, 540+5, 50, 50, '#ffaa00', 20);
                    gc.fillStyle = '#fff'; gc.font = '20px monospace'; gc.fillText(d.game.score, 20, 40);

                    const bw = 800, bh = 600;
                    bc.fillStyle = 'rgba(0,0,0,0.25)'; bc.fillRect(0,0,bw,bh); 
                    const brain = d.brain;
                    const nodes = getNodes(brain.res.length, bw, bh);

                    if (brain.links) {
                        bc.lineWidth = 1;
                        brain.links.forEach(([src, dst]) => {
                            const val = Math.abs(brain.res[src]);
                            if (val > 0.1) {
                                const n1 = nodes[src];
                                const n2 = nodes[dst];
                                bc.beginPath(); bc.moveTo(n1.x, n1.y); bc.lineTo(n2.x, n2.y);
                                bc.strokeStyle = `rgba(255, 170, 0, ${val * 0.4})`;
                                bc.stroke();
                            }
                        });
                    }

                    nodes.forEach((n, i) => {
                        const act = brain.res[i];
                        const size = 3 + Math.abs(act) * 10;
                        const color = act > 0 ? '#ffaa00' : '#ff0055';
                        bc.beginPath(); bc.arc(n.x, n.y, size, 0, 6.28);
                        bc.fillStyle = color; bc.shadowBlur = size * 2; bc.shadowColor = color;
                        bc.fill(); bc.shadowBlur = 0;
                    });

                    brain.vis.forEach((v, i) => {
                        const y = 50 + i * 30;
                        const act = Math.max(0, v);
                        bc.fillStyle = `rgba(255, 255, 255, ${act})`;
                        bc.fillRect(10, y, 5, 5);
                        bc.fillText(`VIS_${i}`, 20, y+5);
                    });

                    brain.out.forEach((v, i) => {
                        const y = 200 + i * 60;
                        drawRect(bc, bw-40, y, 20, 20, `rgba(255, 170, 0, ${v})`, v*20);
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
        'runs': SYSTEM_STATE['runs_completed'],
        'score': SYSTEM_STATE['best_score'],
        'mode': SYSTEM_STATE['mode'],
        'id': SYSTEM_STATE.get('current_id', '---'),
        'logs': SYSTEM_STATE['logs'],
        'game': SYSTEM_STATE['game_view'],
        'brain': SYSTEM_STATE['brain_view'],
        'params': SYSTEM_STATE['hyperparams']
    })

@app.route('/demo_best')
def demo_best():
    if SYSTEM_STATE['best_weights'] is not None:
        SYSTEM_STATE['manual_demo_request'] = True
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'no_champion'})

if __name__ == '__main__':
    engine = TrainingEngine()
    engine.daemon = True
    engine.start()
    print("NEURO-GLASS CONTROL v4 LIVE: http://127.0.0.1:5000")
    print("Fixed Hyperparameters (No Evolution):")
    for k, v in FIXED_PARAMS.items():
        print(f"  {k}: {v}")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)