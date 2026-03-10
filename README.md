# Analog Neural Network Inference

A tiny character-level neural network (3515 parameters) trained in numpy and simulated as a **resistive crossbar array** in ngspice.

## What it does

Given 3 characters, predicts the next character — entirely in analog circuitry.

```
Input: "hel" → Analog circuit predicts: 'l' ✓
Input: "com" → Analog circuit predicts: 'p' ✓
Input: "wor" → Analog circuit predicts: 'l' ✓
```

## Architecture

- **Input**: 81 voltages (3 × 27 one-hot encoded characters)
- **Layer 1**: 81→32 resistive crossbar + diode ReLU (2592 resistors)
- **Layer 2**: 32→27 resistive crossbar + winner-take-all (864 resistors)
- **Total**: 6,912 resistors in the full crossbar version

Matrix multiplication happens via Ohm's law: `V × G = I` (voltage × conductance = current). Kirchhoff's current law sums the products at each column node — instant parallel dot product.

## Files

| File | Purpose |
|------|---------|
| `train_model.py` | Train model in pure numpy, export weights |
| `generate_netlist.py` | Convert weights → ngspice crossbar circuit |
| `run_inference.py` | Run digital vs analog inference, compare |
| `dashboard_nn.py` | Interactive Plotly dashboard |
| `weights.json` | Trained model weights |
| `weights_analog.json` | Weights as conductance values |

## Quick Start

```bash
# Train (or use pre-trained weights.json)
python3 train_model.py

# Generate ngspice circuit for input "hel"
python3 generate_netlist.py --input "hel"

# Run and compare digital vs analog
python3 run_inference.py --save

# Generate interactive dashboard
python3 dashboard_nn.py
```

## Requirements

- Python 3.8+ with numpy
- ngspice (for analog simulation)
- plotly (for dashboard)
