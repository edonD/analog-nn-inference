#!/usr/bin/env python3
"""
Generate a visual HTML dashboard showing:
  1. Input digit images (actual rendered pixels)
  2. The analog crossbar circuit schematic
  3. Output voltage bar charts per digit
  4. Autoresearch optimization history
  5. Full accuracy confusion matrix

Usage:
    python dashboard.py                       # Generate dashboard from eval_results.json
    python dashboard.py --run-demo 10         # Run 10 digits through ngspice first, then dashboard
    python dashboard.py --output report.html  # Custom output path
"""

import argparse
import json
import os
import sys
import subprocess
import re
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def load_data():
    sys.path.insert(0, str(SCRIPT_DIR))
    from train_model import load_dataset, train_test_split
    X, y = load_dataset()
    _, _, X_test, y_test = train_test_split(X, y)
    with open(SCRIPT_DIR / "weights.json", "r") as f:
        w = json.load(f)
    return X_test, y_test, w


def digital_forward(x, W1, b1, W2, b2):
    z1 = x @ np.array(W1) + np.array(b1)
    h1 = np.maximum(0, z1)
    logits = h1 @ np.array(W2) + np.array(b2)
    return h1, logits


def run_analog_single(pixels, label, ngspice_cmd):
    """Run one digit through ngspice, return output voltages."""
    pixel_str = ",".join(f"{v:.6f}" for v in pixels)
    gen_cmd = [sys.executable, str(SCRIPT_DIR / "generate_circuit.py"),
               "--pixel-values", pixel_str, "--label", str(label), "--mode", "crossbar"]
    subprocess.run(gen_cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPT_DIR))

    circuit_file = str(SCRIPT_DIR / "analog_mnist.cir")
    output_file = str(SCRIPT_DIR / "analog_output.txt")
    if os.path.exists(output_file):
        os.remove(output_file)

    t0 = time.time()
    result = subprocess.run([ngspice_cmd, "-b", circuit_file],
                            capture_output=True, text=True, timeout=120, cwd=str(SCRIPT_DIR))
    sim_time = time.time() - t0

    outputs = None
    if os.path.isfile(output_file):
        with open(output_file, "r") as f:
            content = f.read().strip()
        if content:
            nums = []
            for t in content.split():
                try:
                    nums.append(float(t))
                except ValueError:
                    pass
            if len(nums) >= 20:
                outputs = [nums[i * 2 + 1] for i in range(10)]

    if outputs is None:
        pattern = re.compile(r"out_(\d+)\s*(?:\([^)]*\))?\s*=\s*([-\d.eE+]+)")
        values = {}
        for line in result.stdout.split("\n"):
            m = pattern.search(line)
            if m:
                values[int(m.group(1))] = float(m.group(2))
        if len(values) >= 10:
            outputs = [values.get(i, 0.0) for i in range(10)]

    return outputs, sim_time


def load_eval_results():
    path = SCRIPT_DIR / "eval_results.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_history():
    path = SCRIPT_DIR / "experiment_history.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"experiments": []}


def pixel_to_color(val):
    """Convert 0-1 pixel value to hex color."""
    v = int(val * 255)
    return f"rgb({v},{v},{v})"


def generate_html(demo_results, weights_data, history, eval_data):
    """Generate the full HTML dashboard."""

    W1 = np.array(weights_data["W1"])
    b1 = np.array(weights_data["b1"])
    W2 = np.array(weights_data["W2"])
    b2 = np.array(weights_data["b2"])
    arch = weights_data.get("architecture", {})
    n_hidden = len(b1)
    n_in = W1.shape[0]
    n_out = W2.shape[1]

    # Weight stats
    w1_max = float(np.abs(W1).max())
    w2_max = float(np.abs(W2).max())
    total_params = W1.size + b1.size + W2.size + b2.size
    total_resistors = 2 * (n_in * n_hidden + n_hidden * n_out)

    # Build demo results JSON for JS
    demo_json = json.dumps(demo_results)
    history_json = json.dumps(history.get("experiments", []))

    # Build W1 heatmap data (downsample for display)
    w1_sample = W1[:, :n_hidden].tolist()
    w2_sample = W2.tolist()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Analog MNIST — A Circuit That Sees</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0a0a0f; color: #e0e0e0;
    line-height: 1.6;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}

  /* Header */
  .header {{
    text-align: center; padding: 40px 20px;
    background: linear-gradient(135deg, #0a0a2e 0%, #1a0a3e 50%, #0a2a2e 100%);
    border-bottom: 2px solid #333;
  }}
  .header h1 {{ font-size: 2.5em; color: #00ff88; margin-bottom: 10px; }}
  .header .subtitle {{ color: #888; font-size: 1.1em; }}
  .header .stats {{
    display: flex; justify-content: center; gap: 40px; margin-top: 20px;
    flex-wrap: wrap;
  }}
  .stat-box {{
    background: rgba(255,255,255,0.05); border: 1px solid #333;
    border-radius: 8px; padding: 15px 25px; text-align: center;
  }}
  .stat-box .number {{ font-size: 2em; font-weight: bold; color: #00ff88; }}
  .stat-box .label {{ font-size: 0.85em; color: #888; }}

  /* Sections */
  .section {{
    margin: 30px 0; padding: 25px;
    background: #111118; border: 1px solid #222;
    border-radius: 12px;
  }}
  .section h2 {{
    color: #00cc66; font-size: 1.4em; margin-bottom: 15px;
    border-bottom: 1px solid #222; padding-bottom: 8px;
  }}
  .section h3 {{ color: #aaa; font-size: 1.1em; margin: 15px 0 8px; }}

  /* Circuit schematic */
  .schematic {{
    display: flex; justify-content: center; padding: 20px;
  }}
  .schematic svg {{ max-width: 100%; }}

  /* Digit demo cards */
  .demo-grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 20px;
  }}
  .demo-card {{
    background: #0d0d15; border: 1px solid #2a2a3a;
    border-radius: 10px; padding: 18px; position: relative;
  }}
  .demo-card.correct {{ border-color: #00ff8840; }}
  .demo-card.wrong {{ border-color: #ff444440; }}
  .demo-card .card-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
  }}
  .demo-card .verdict {{
    font-weight: bold; font-size: 0.9em; padding: 3px 10px;
    border-radius: 4px;
  }}
  .demo-card .verdict.ok {{ background: #00ff8820; color: #00ff88; }}
  .demo-card .verdict.fail {{ background: #ff444420; color: #ff4444; }}

  .card-body {{ display: flex; gap: 15px; align-items: flex-start; }}

  /* Pixel grid */
  .pixel-grid {{
    display: grid; grid-template-columns: repeat(8, 1fr);
    gap: 1px; width: 120px; height: 120px; flex-shrink: 0;
    border: 2px solid #333; border-radius: 4px; overflow: hidden;
  }}
  .pixel {{
    width: 100%; aspect-ratio: 1;
  }}

  /* Voltage bars */
  .voltage-bars {{ flex: 1; }}
  .v-bar {{
    display: flex; align-items: center; margin: 2px 0;
    font-size: 0.75em; font-family: monospace;
  }}
  .v-bar .digit-label {{ width: 14px; text-align: right; margin-right: 5px; color: #888; }}
  .v-bar .bar-bg {{
    flex: 1; height: 14px; background: #1a1a2a;
    border-radius: 2px; overflow: hidden; position: relative;
  }}
  .v-bar .bar-fill {{
    height: 100%; border-radius: 2px;
    transition: width 0.3s;
  }}
  .v-bar .bar-fill.winner {{ background: linear-gradient(90deg, #00aa55, #00ff88); }}
  .v-bar .bar-fill.normal {{ background: #334; }}
  .v-bar .bar-fill.negative {{ background: #442233; }}
  .v-bar .v-value {{ width: 55px; text-align: right; margin-left: 4px; color: #666; font-size: 0.85em; }}

  /* Weight heatmap */
  .heatmap-container {{
    overflow-x: auto; padding: 10px 0;
  }}
  .heatmap {{
    display: grid; gap: 0;
  }}
  .heatmap-cell {{
    width: 100%; aspect-ratio: 1;
  }}

  /* History chart */
  .history-chart {{
    width: 100%; height: 200px; position: relative;
    background: #0d0d15; border: 1px solid #222; border-radius: 8px;
    margin-top: 10px;
  }}

  /* Crossbar visual */
  .crossbar-visual {{
    display: flex; flex-direction: column; align-items: center;
    padding: 20px;
  }}
  .crossbar-row {{
    display: flex; align-items: center; gap: 2px;
  }}
  .crossbar-cell {{
    width: 6px; height: 6px; border-radius: 1px;
  }}
  .crossbar-label {{
    font-size: 0.7em; color: #666; width: 30px; text-align: right;
    margin-right: 5px;
  }}

  /* Confusion matrix */
  .confusion-table {{
    border-collapse: collapse; margin: 10px auto;
  }}
  .confusion-table td, .confusion-table th {{
    width: 36px; height: 36px; text-align: center;
    font-size: 0.8em; font-family: monospace;
    border: 1px solid #222;
  }}
  .confusion-table th {{ color: #00cc66; background: #0a0a15; }}

  .flow-diagram {{
    display: flex; align-items: center; justify-content: center;
    gap: 10px; flex-wrap: wrap; padding: 20px 0;
  }}
  .flow-box {{
    background: #1a1a2e; border: 2px solid #333;
    border-radius: 10px; padding: 12px 18px;
    text-align: center; min-width: 140px;
  }}
  .flow-box.input {{ border-color: #4488ff; }}
  .flow-box.layer {{ border-color: #ff8844; }}
  .flow-box.activation {{ border-color: #ffcc00; }}
  .flow-box.output {{ border-color: #00ff88; }}
  .flow-box .box-title {{ font-weight: bold; font-size: 0.95em; }}
  .flow-box .box-detail {{ font-size: 0.75em; color: #888; margin-top: 4px; }}
  .flow-arrow {{ color: #444; font-size: 1.5em; }}

  .inline-note {{
    background: #1a1a2e; border-left: 3px solid #00cc66;
    padding: 12px 16px; margin: 12px 0; border-radius: 0 6px 6px 0;
    font-size: 0.9em; color: #aaa;
  }}
  .inline-note strong {{ color: #00ff88; }}
</style>
</head>
<body>

<div class="header">
  <h1>Analog MNIST</h1>
  <div class="subtitle">A resistive crossbar circuit that classifies handwritten digits through pure physics</div>
  <div class="stats">
    <div class="stat-box">
      <div class="number">{total_resistors:,}</div>
      <div class="label">Resistors</div>
    </div>
    <div class="stat-box">
      <div class="number">{n_hidden}</div>
      <div class="label">Diodes (ReLU)</div>
    </div>
    <div class="stat-box">
      <div class="number">{total_params:,}</div>
      <div class="label">Trained Weights</div>
    </div>
    <div class="stat-box">
      <div class="number" id="accuracy-stat">—</div>
      <div class="label">Analog Accuracy</div>
    </div>
    <div class="stat-box">
      <div class="number">0.04s</div>
      <div class="label">Per Inference</div>
    </div>
  </div>
</div>

<div class="container">

<!-- Section 1: How it works -->
<div class="section">
  <h2>How It Works — The Physics of Neural Network Inference</h2>

  <div class="flow-diagram">
    <div class="flow-box input">
      <div class="box-title">8&times;8 Image</div>
      <div class="box-detail">64 pixels &rarr; 64 voltages<br>0V (black) to 1V (white)</div>
    </div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-box layer">
      <div class="box-title">Crossbar Layer 1</div>
      <div class="box-detail">64&times;{n_hidden} = {2*n_in*n_hidden:,} resistors<br>Ohm: V/R=I &nbsp; KCL: &Sigma;I=0</div>
    </div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-box activation">
      <div class="box-title">{n_hidden} Diode ReLU</div>
      <div class="box-detail">Pass positive voltage<br>Block negative voltage</div>
    </div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-box layer">
      <div class="box-title">Crossbar Layer 2</div>
      <div class="box-detail">{n_hidden}&times;10 = {2*n_hidden*n_out} resistors<br>Same physics, smaller</div>
    </div>
    <div class="flow-arrow">&rarr;</div>
    <div class="flow-box output">
      <div class="box-title">10 Outputs</div>
      <div class="box-detail">One voltage per digit<br>Highest wins</div>
    </div>
  </div>

  <div class="inline-note">
    <strong>No code executes during inference.</strong> The voltages physically flow through resistors.
    Ohm's law does multiplication (V &times; G = I), Kirchhoff's current law does addition (&Sigma;I = 0),
    and diodes perform ReLU activation (pass positive, block negative).
    The entire neural network forward pass resolves as a single DC operating point — one "tick" of physics.
  </div>
</div>

<!-- Section 2: Circuit Schematic -->
<div class="section">
  <h2>Circuit Schematic — Resistive Crossbar Array</h2>
  <p style="color:#888; font-size:0.9em; margin-bottom:15px;">
    Each dot below is a resistor. Brighter = larger weight magnitude.
    Every weight is stored as a differential pair (R+, R-) of physical resistors.
  </p>

  <h3>Layer 1: Input (64 pixels) &rarr; Hidden ({n_hidden} neurons)</h3>
  <div class="crossbar-visual" id="crossbar-l1"></div>

  <h3>Layer 2: Hidden ({n_hidden} neurons) &rarr; Output (10 digits)</h3>
  <div class="crossbar-visual" id="crossbar-l2"></div>

  <div class="inline-note">
    <strong>How to read:</strong> Each row is an input (pixel or hidden neuron). Each column is an output neuron.
    Each intersection has a resistor whose value encodes the corresponding weight.
    Bright green = large positive weight. Bright red = large negative weight. Dark = near-zero weight.
  </div>
</div>

<!-- Section 3: Weight Heatmaps -->
<div class="section">
  <h2>Trained Weight Matrices</h2>
  <p style="color:#888; font-size:0.9em; margin-bottom:10px;">
    These weights were trained digitally (numpy), then mapped to physical conductance values.
    W1 max: {w1_max:.3f}, W2 max: {w2_max:.3f}
  </p>
  <div style="display:flex; gap:30px; flex-wrap:wrap;">
    <div>
      <h3>W1 ({n_in}&times;{n_hidden}) — Pixel &rarr; Hidden</h3>
      <canvas id="w1-canvas" width="640" height="320" style="border:1px solid #333; border-radius:6px;"></canvas>
    </div>
    <div>
      <h3>W2 ({n_hidden}&times;{n_out}) — Hidden &rarr; Digit</h3>
      <canvas id="w2-canvas" width="320" height="320" style="border:1px solid #333; border-radius:6px;"></canvas>
    </div>
  </div>
</div>

<!-- Section 4: Live Demo Results -->
<div class="section">
  <h2>Live Inference — Digits Through the Circuit</h2>
  <p style="color:#888; font-size:0.9em; margin-bottom:15px;">
    Each card shows a test digit fed through the analog circuit in ngspice.
    Left: the 8&times;8 input image. Right: output voltage per digit class (highest = prediction).
  </p>
  <div class="demo-grid" id="demo-grid"></div>
</div>

<!-- Section 5: Autoresearch History -->
<div class="section">
  <h2>Autoresearch Optimization Journey</h2>
  <p style="color:#888; font-size:0.9em; margin-bottom:15px;">
    The autoresearcher iterates on analog design parameters to maximize classification accuracy.
  </p>
  <canvas id="history-canvas" width="1000" height="250" style="width:100%; border:1px solid #222; border-radius:8px; background:#0d0d15;"></canvas>
  <div id="history-table" style="margin-top:15px;"></div>
</div>

<!-- Section 6: Confusion Matrix -->
<div class="section">
  <h2>Accuracy Breakdown</h2>
  <div style="display:flex; gap:30px; flex-wrap:wrap; align-items:flex-start;">
    <div>
      <h3>Analog Confusion Matrix</h3>
      <table class="confusion-table" id="confusion-matrix"></table>
    </div>
    <div id="per-digit-accuracy" style="flex:1; min-width:300px;"></div>
  </div>
</div>

</div><!-- container -->

<div style="text-align:center; padding:30px; color:#444; font-size:0.85em; border-top:1px solid #222; margin-top:30px;">
  Generated by Analog MNIST Autoresearcher &mdash;
  <a href="https://github.com/edonD/analog-nn-inference" style="color:#00cc66;">GitHub</a>
  &mdash; Each inference runs {total_resistors:,} resistors + {n_hidden} diodes in ngspice
</div>

<script>
const demoResults = {demo_json};
const historyData = {history_json};
const W1 = {json.dumps(w1_sample)};
const W2 = {json.dumps(w2_sample)};
const W1_MAX = {w1_max};
const W2_MAX = {w2_max};
const N_IN = {n_in};
const N_HIDDEN = {n_hidden};
const N_OUT = {n_out};

// --- Render crossbar visualizations ---
function renderCrossbar(containerId, weights, maxW) {{
  const container = document.getElementById(containerId);
  const rows = weights.length;
  const cols = weights[0].length;

  for (let i = 0; i < rows; i++) {{
    const row = document.createElement('div');
    row.className = 'crossbar-row';

    if (i % 8 === 0 || rows <= 32) {{
      const label = document.createElement('span');
      label.className = 'crossbar-label';
      label.textContent = i;
      row.appendChild(label);
    }} else {{
      const spacer = document.createElement('span');
      spacer.className = 'crossbar-label';
      row.appendChild(spacer);
    }}

    for (let j = 0; j < cols; j++) {{
      const cell = document.createElement('div');
      cell.className = 'crossbar-cell';
      const w = weights[i][j];
      const intensity = Math.min(Math.abs(w) / maxW, 1.0);
      const alpha = 0.15 + intensity * 0.85;
      if (w > 0) {{
        cell.style.background = `rgba(0, 255, 100, ${{alpha}})`;
      }} else {{
        cell.style.background = `rgba(255, 60, 60, ${{alpha}})`;
      }}
      cell.title = `[${{i}}][${{j}}] = ${{w.toFixed(4)}}`;
      row.appendChild(cell);
    }}
    container.appendChild(row);
  }}
}}
renderCrossbar('crossbar-l1', W1, W1_MAX);
renderCrossbar('crossbar-l2', W2, W2_MAX);

// --- Render weight heatmaps on canvas ---
function renderHeatmap(canvasId, weights, maxW) {{
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const rows = weights.length;
  const cols = weights[0].length;
  const cellW = canvas.width / cols;
  const cellH = canvas.height / rows;

  for (let i = 0; i < rows; i++) {{
    for (let j = 0; j < cols; j++) {{
      const w = weights[i][j];
      const norm = w / maxW; // -1 to 1
      let r, g, b;
      if (norm > 0) {{
        r = 0; g = Math.floor(norm * 255); b = Math.floor(norm * 100);
      }} else {{
        r = Math.floor(-norm * 255); g = 0; b = Math.floor(-norm * 100);
      }}
      ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
      ctx.fillRect(j * cellW, i * cellH, cellW, cellH);
    }}
  }}
}}
renderHeatmap('w1-canvas', W1, W1_MAX);
renderHeatmap('w2-canvas', W2, W2_MAX);

// --- Render demo cards ---
function renderDemoCards() {{
  const grid = document.getElementById('demo-grid');
  let correct = 0;
  let total = 0;

  demoResults.forEach((r, idx) => {{
    if (!r.analog_outputs) return;
    total++;
    const isCorrect = r.analog_pred === r.true_label;
    if (isCorrect) correct++;

    const card = document.createElement('div');
    card.className = `demo-card ${{isCorrect ? 'correct' : 'wrong'}}`;

    // Header
    const header = document.createElement('div');
    header.className = 'card-header';
    header.innerHTML = `
      <span style="color:#888">Test #${{r.index}} &mdash; True: <b style="color:#fff; font-size:1.3em">${{r.true_label}}</b></span>
      <span class="verdict ${{isCorrect ? 'ok' : 'fail'}}">${{isCorrect ? '✓ CORRECT' : '✗ WRONG'}}</span>
    `;
    card.appendChild(header);

    // Body: pixel grid + voltage bars
    const body = document.createElement('div');
    body.className = 'card-body';

    // Pixel grid
    const pixelGrid = document.createElement('div');
    pixelGrid.className = 'pixel-grid';
    const pixels = r.pixels || [];
    for (let i = 0; i < 64; i++) {{
      const px = document.createElement('div');
      px.className = 'pixel';
      const v = pixels[i] || 0;
      const c = Math.floor(v * 255);
      px.style.background = `rgb(${{c}},${{c}},${{c}})`;
      pixelGrid.appendChild(px);
    }}
    body.appendChild(pixelGrid);

    // Voltage bars
    const bars = document.createElement('div');
    bars.className = 'voltage-bars';
    const outputs = r.analog_outputs;
    const maxV = Math.max(...outputs.map(v => Math.abs(v)));

    for (let d = 0; d < 10; d++) {{
      const v = outputs[d];
      const bar = document.createElement('div');
      bar.className = 'v-bar';

      const isWinner = d === r.analog_pred;
      const pct = Math.max(0, (v + maxV) / (2 * maxV) * 100);
      const fillClass = isWinner ? 'winner' : (v >= 0 ? 'normal' : 'negative');

      bar.innerHTML = `
        <span class="digit-label">${{d}}</span>
        <div class="bar-bg">
          <div class="bar-fill ${{fillClass}}" style="width:${{pct}}%"></div>
        </div>
        <span class="v-value">${{v >= 0 ? '+' : ''}}${{v.toFixed(1)}}V</span>
      `;
      bars.appendChild(bar);
    }}
    body.appendChild(bars);
    card.appendChild(body);

    // Footer: sim time + cosine similarity
    const footer = document.createElement('div');
    footer.style.cssText = 'margin-top:8px; font-size:0.8em; color:#666; display:flex; justify-content:space-between;';
    const cosSim = r.cosine_sim ? r.cosine_sim.toFixed(4) : '—';
    const simTime = r.sim_time ? r.sim_time.toFixed(2) + 's' : '—';
    footer.innerHTML = `
      <span>Analog: <b style="color:${{isCorrect ? '#00ff88' : '#ff4444'}}">${{r.analog_pred}}</b> &nbsp; Digital: <b>${{r.digital_pred}}</b></span>
      <span>cos=${{cosSim}} &nbsp; ${{simTime}}</span>
    `;
    card.appendChild(footer);

    grid.appendChild(card);
  }});

  // Update accuracy stat
  if (total > 0) {{
    document.getElementById('accuracy-stat').textContent =
      Math.round(correct / total * 100) + '%';
  }}
}}
renderDemoCards();

// --- Render history chart ---
function renderHistory() {{
  const canvas = document.getElementById('history-canvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const pad = {{ top: 30, right: 20, bottom: 30, left: 60 }};

  if (historyData.length === 0) {{
    ctx.fillStyle = '#444';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No experiments yet — run the autoresearcher!', W/2, H/2);
    return;
  }}

  const scores = historyData.map(e => e.scores?.total_score || 0);
  const maxScore = Math.max(...scores, 1);
  const minScore = Math.min(...scores, 0);

  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  // Axes
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, H - pad.bottom);
  ctx.lineTo(W - pad.right, H - pad.bottom);
  ctx.stroke();

  // Y-axis labels
  ctx.fillStyle = '#666';
  ctx.font = '11px monospace';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {{
    const val = minScore + (maxScore - minScore) * i / 4;
    const y = H - pad.bottom - (i / 4) * plotH;
    ctx.fillText(val.toFixed(2), pad.left - 8, y + 4);
    ctx.strokeStyle = '#1a1a2a';
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();
  }}

  // Title
  ctx.fillStyle = '#888';
  ctx.font = '12px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Total Score per Experiment', W / 2, 18);

  // Plot line
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 2;
  ctx.beginPath();
  scores.forEach((s, i) => {{
    const x = pad.left + (i / Math.max(scores.length - 1, 1)) * plotW;
    const y = H - pad.bottom - ((s - minScore) / (maxScore - minScore || 1)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }});
  ctx.stroke();

  // Points
  scores.forEach((s, i) => {{
    const x = pad.left + (i / Math.max(scores.length - 1, 1)) * plotW;
    const y = H - pad.bottom - ((s - minScore) / (maxScore - minScore || 1)) * plotH;
    const isBest = historyData[i]?.is_best;
    ctx.fillStyle = isBest ? '#ffcc00' : '#00ff88';
    ctx.beginPath();
    ctx.arc(x, y, isBest ? 5 : 3, 0, Math.PI * 2);
    ctx.fill();
  }});

  // History table
  const table = document.getElementById('history-table');
  if (historyData.length > 0) {{
    let html = '<table style="width:100%; border-collapse:collapse; font-size:0.85em;">';
    html += '<tr style="color:#00cc66; border-bottom:1px solid #333;">';
    html += '<th style="padding:6px; text-align:left;">#</th>';
    html += '<th style="padding:6px; text-align:left;">Score</th>';
    html += '<th style="padding:6px; text-align:left;">Analog Acc</th>';
    html += '<th style="padding:6px; text-align:left;">Match</th>';
    html += '<th style="padding:6px; text-align:left;">CosSim</th>';
    html += '<th style="padding:6px; text-align:left;">Time</th>';
    html += '<th style="padding:6px; text-align:left;">Notes</th>';
    html += '</tr>';

    historyData.forEach(exp => {{
      const s = exp.scores || {{}};
      const best = exp.is_best ? ' style="color:#ffcc00;"' : '';
      html += `<tr style="border-bottom:1px solid #1a1a2a;">`;
      html += `<td style="padding:4px 6px;"${{best}}>${{exp.id}}</td>`;
      html += `<td style="padding:4px 6px;"${{best}}>${{(s.total_score || 0).toFixed(4)}}</td>`;
      html += `<td style="padding:4px 6px;">${{((s.analog_accuracy || 0) * 100).toFixed(1)}}%</td>`;
      html += `<td style="padding:4px 6px;">${{((s.match_rate || 0) * 100).toFixed(1)}}%</td>`;
      html += `<td style="padding:4px 6px;">${{(s.avg_cosine_sim || 0).toFixed(4)}}</td>`;
      html += `<td style="padding:4px 6px;">${{exp.timestamp || ''}}</td>`;
      html += `<td style="padding:4px 6px; color:#666;">${{exp.notes || ''}}</td>`;
      html += '</tr>';
    }});
    html += '</table>';
    table.innerHTML = html;
  }}
}}
renderHistory();

// --- Confusion matrix ---
function renderConfusion() {{
  const table = document.getElementById('confusion-matrix');
  const cm = Array(10).fill(null).map(() => Array(10).fill(0));
  let total = 0;

  demoResults.forEach(r => {{
    if (r.analog_outputs && r.true_label !== undefined && r.analog_pred !== undefined) {{
      cm[r.true_label][r.analog_pred]++;
      total++;
    }}
  }});

  let html = '<tr><th></th>';
  for (let j = 0; j < 10; j++) html += `<th>${{j}}</th>`;
  html += '</tr>';

  const maxVal = Math.max(...cm.flat(), 1);
  for (let i = 0; i < 10; i++) {{
    html += `<tr><th>${{i}}</th>`;
    for (let j = 0; j < 10; j++) {{
      const v = cm[i][j];
      let bg = '#0a0a15';
      if (v > 0) {{
        if (i === j) {{
          const alpha = 0.2 + 0.8 * v / maxVal;
          bg = `rgba(0, 255, 100, ${{alpha}})`;
        }} else {{
          const alpha = 0.3 + 0.7 * v / maxVal;
          bg = `rgba(255, 60, 60, ${{alpha}})`;
        }}
      }}
      html += `<td style="background:${{bg}}">${{v || ''}}</td>`;
    }}
    html += '</tr>';
  }}
  table.innerHTML = html;

  // Per-digit accuracy
  const div = document.getElementById('per-digit-accuracy');
  let accHtml = '<h3>Per-Digit Accuracy</h3>';
  for (let d = 0; d < 10; d++) {{
    const rowSum = cm[d].reduce((a, b) => a + b, 0);
    const correct = cm[d][d];
    const pct = rowSum > 0 ? (correct / rowSum * 100) : 0;
    const barW = pct;
    accHtml += `<div style="display:flex; align-items:center; margin:4px 0; font-size:0.9em;">
      <span style="width:20px; color:#888; text-align:right; margin-right:8px;">${{d}}</span>
      <div style="flex:1; height:18px; background:#1a1a2a; border-radius:3px; overflow:hidden;">
        <div style="width:${{barW}}%; height:100%; background:${{pct >= 80 ? '#00cc66' : pct >= 50 ? '#ffcc00' : '#ff4444'}}; border-radius:3px;"></div>
      </div>
      <span style="width:60px; text-align:right; margin-left:8px; font-family:monospace; color:${{pct >= 80 ? '#00cc66' : '#ff4444'}}">${{correct}}/${{rowSum}}</span>
    </div>`;
  }}
  div.innerHTML = accHtml;
}}
renderConfusion();
</script>
</body>
</html>"""
    return html


def run_demo_digits(n_digits, seed=42):
    """Run n digits through analog circuit and return results."""
    X_test, y_test, weights_data = load_data()

    # Find ngspice
    ngspice_cmd = None
    for candidate in ["ngspice", "/usr/local/bin/ngspice", "/usr/bin/ngspice"]:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
            ngspice_cmd = candidate
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    W1 = np.array(weights_data["W1"])
    b1 = np.array(weights_data["b1"])
    W2 = np.array(weights_data["W2"])
    b2 = np.array(weights_data["b2"])

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X_test), size=min(n_digits, len(X_test)), replace=False)

    results = []
    for idx in indices:
        idx = int(idx)
        pixels = X_test[idx]
        label = int(y_test[idx])

        # Digital
        _, d_logits = digital_forward(pixels, weights_data["W1"], weights_data["b1"],
                                       weights_data["W2"], weights_data["b2"])
        d_pred = int(np.argmax(d_logits))

        r = {
            "index": idx,
            "true_label": label,
            "pixels": pixels.tolist(),
            "digital_pred": d_pred,
            "digital_logits": d_logits.tolist(),
        }

        # Analog
        if ngspice_cmd:
            outputs, sim_time = run_analog_single(pixels, label, ngspice_cmd)
            if outputs:
                a_pred = int(np.argmax(outputs))
                r["analog_outputs"] = outputs
                r["analog_pred"] = a_pred
                r["sim_time"] = sim_time
                norm_a = np.linalg.norm(outputs)
                norm_d = np.linalg.norm(d_logits)
                if norm_a > 0 and norm_d > 0:
                    r["cosine_sim"] = float(np.dot(outputs, d_logits) / (norm_a * norm_d))

                status = "OK" if a_pred == label else "WRONG"
                print(f"  [{len(results)+1:3d}/{n_digits}] digit={label} analog={a_pred} [{status}] {sim_time:.2f}s")
            else:
                print(f"  [{len(results)+1:3d}/{n_digits}] digit={label} PARSE ERROR")
        else:
            print(f"  [{len(results)+1:3d}/{n_digits}] digit={label} (no ngspice)")

        results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate visual HTML dashboard.")
    parser.add_argument("--run-demo", type=int, default=0,
                        help="Run N digits through ngspice before generating dashboard")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_test, y_test, weights_data = load_data()
    history = load_history()
    eval_data = load_eval_results()

    if args.run_demo > 0:
        print(f"Running {args.run_demo} digits through analog circuit...")
        demo_results = run_demo_digits(args.run_demo, seed=args.seed)
    elif eval_data and "results" in eval_data:
        # Use existing eval results, add pixel data
        demo_results = eval_data["results"]
        for r in demo_results:
            idx = r.get("index", 0)
            if idx < len(X_test):
                r["pixels"] = X_test[idx].tolist()
    else:
        # Run a small demo
        print("No eval results found. Running 10 digits...")
        demo_results = run_demo_digits(10, seed=args.seed)

    html = generate_html(demo_results, weights_data, history, eval_data)

    out_path = args.output or str(SCRIPT_DIR / "dashboard.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nDashboard saved to: {out_path}")
    print(f"Open in browser to view.")


if __name__ == "__main__":
    main()
