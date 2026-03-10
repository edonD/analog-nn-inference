# Analog MNIST Autoresearch Program

## Mission

Build an analog circuit that classifies handwritten digits (0-9) using resistive crossbar arrays. The circuit performs neural network inference through pure analog physics — Ohm's law for multiplication, Kirchhoff's current law for summation, and diodes for ReLU activation.

The goal: **achieve >= 80% analog classification accuracy** on 50 test digits from the sklearn 8x8 digits dataset.

## How It Works

### The Circuit

```
8x8 digit image (64 pixels)
         ↓
64 voltage sources (0-1V each, proportional to pixel intensity)
         ↓
┌──────────────────────────────────────┐
│  LAYER 1: 64×32 Resistive Crossbar  │
│  - 4,096 resistors (differential)    │
│  - Virtual ground summing nodes       │
│  - Transimpedance readout            │
└──────────────┬───────────────────────┘
               ↓
    32 Diode ReLU circuits (D + R_pulldown)
               ↓
┌──────────────────────────────────────┐
│  LAYER 2: 32×10 Resistive Crossbar  │
│  - 640 resistors (differential)      │
└──────────────┬───────────────────────┘
               ↓
    10 output voltages → highest = predicted digit
```

### The Physics

Each weight W[i][j] maps to a **differential conductance pair**:
- G+ = G_offset + W × G_scale/2
- G- = G_offset - W × G_scale/2
- Effective current: I = V_in × (G+ - G-) = V_in × W × G_scale

Kirchhoff's current law sums all contributions at the column node:
- I_total = Σ V_in[i] × W[i][j] × G_scale = dot_product × G_scale

A transimpedance stage converts current to voltage:
- V_out = I_total × R_tia + bias

This is **exact matrix multiplication** in physics — no digital computation.

## Experiment Loop

### Step 1: Check Status
```bash
python3 autoresearch.py status
```

### Step 2: Get Suggestion
```bash
python3 autoresearch.py suggest
```
This writes `suggested_params.json` with recommended parameter changes.

### Step 3: Apply Parameters and Evaluate
```bash
python3 evaluate.py --params suggested_params.json --n-test 50 2>&1 | tee run.log
```

### Step 4: Log Results
```bash
python3 autoresearch.py log --notes "description of what changed"
```

### Step 5: If Score Improved → Commit & Push
```bash
# On dev branch
git add -A
git commit -m "exp: score X.XXX -- description of changes"
git push origin dev

# Create PR
gh pr create --base main --head dev \
  --title "[MNIST] score: X.XX -- description" \
  --body "## Metrics
- Analog accuracy: XX.X%
- Digital-analog match: XX.X%
- Cosine similarity: X.XXXX
- Total score: X.XXXX

## Changes
- description of parameter changes

## Next
- what to try next"

# Merge and pull
gh pr merge --merge --delete-branch=false
git checkout main && git pull && git checkout dev && git merge main
```

### Step 6: Repeat

NEVER stop iterating. If the score plateaus:
1. Try different parameter regions (G_scale, diode_n are most impactful)
2. Change hidden layer size (retrain with `python3 train_model.py --hidden 64`)
3. Try hybrid mode: `python3 evaluate.py --mode hybrid` (behavioral matmul)
4. Implement circuit improvements:
   - Opamp-based ReLU (sharper activation, no diode voltage drop)
   - Voltage scaling between layers (compensate for signal attenuation)
   - Weight quantization-aware training (train with limited precision)

## Scoring

Score = 0.6 × analog_accuracy + 0.3 × match_rate + 0.1 × cosine_similarity

- **analog_accuracy**: fraction of test digits the circuit classifies correctly
- **match_rate**: fraction where analog and digital predictions agree
- **cosine_similarity**: fidelity of analog output vector vs digital logits

## Parameter Guide

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| G_scale | Higher → stronger signals but more nonlinearity | 1e-4 to 1e-2 |
| diode_n | Lower → sharper ReLU, but harder convergence | 0.001 to 0.1 |
| diode_is | Affects diode turn-on behavior | 1e-18 to 1e-12 |
| R_tia | Higher → more gain per column | 100 to 10k |
| R_pulldown | Higher → less leakage but slower | 100k to 10M |
| V_high | Higher → stronger input signals | 0.5 to 3.3 |

## Architecture Changes

To change hidden layer size:
```bash
python3 train_model.py --hidden 64 --epochs 2000
# This retrains and exports new weights.json
# Then re-evaluate:
python3 evaluate.py --n-test 50
```

## Files

| File | Purpose | Modify? |
|------|---------|---------|
| train_model.py | Train digital MLP, export weights | YES (architecture) |
| generate_circuit.py | Convert weights → ngspice crossbar | YES (circuit techniques) |
| evaluate.py | Score analog accuracy | NO |
| autoresearch.py | Experiment management | NO |
| weights.json | Trained model weights | AUTO-GENERATED |
| suggested_params.json | Current analog parameters | AUTO-GENERATED |
| eval_results.json | Latest evaluation results | AUTO-GENERATED |
| experiment_history.json | All experiment history | AUTO-GENERATED |

## Anti-Gaming Rules

- evaluate.py parses the circuit and validates outputs independently
- Accuracy is measured on held-out test digits not seen during training
- The circuit must use real ngspice simulation (no hardcoded outputs)
- Cosine similarity catches cases where the circuit "cheats" by scaling

## What Makes This Impressive

This isn't a toy. This is how **real analog AI accelerators work**:
- Mythic, Syntiant, analog-ml chips use resistive crossbars
- Each resistor stores a weight (like a memristor/ReRAM cell)
- Inference happens in **one clock cycle** — O(1) regardless of matrix size
- Power consumption is orders of magnitude lower than digital
- We're doing the same thing, but proving it works in SPICE first
