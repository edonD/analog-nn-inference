# Analog MNIST Autoresearch Project

An analog circuit that classifies handwritten digits. Physical resistors do the math.

Read `program.md` for full instructions. This is your bible.

## Quick Start
1. Read `program.md` completely
2. Run `python3 train_model.py` to train the digital model + export weights
3. Run `python3 evaluate.py --n-test 20` for baseline score
4. Run `python3 autoresearch.py baseline` to log baseline
5. Start the optimization loop (see program.md)

## Key Commands
```bash
# Train digital model (only needed once, or when changing hidden_dim)
python3 train_model.py --hidden 32 --epochs 1500

# Generate circuit for a single digit
python3 generate_circuit.py --digit-index 0

# Run evaluation (score the analog design)
python3 evaluate.py --n-test 50 --mode crossbar 2>&1 | tee run.log

# Autoresearcher commands
python3 autoresearch.py status      # Current best score
python3 autoresearch.py suggest     # Get next parameter suggestion
python3 autoresearch.py log         # Log latest eval results
python3 autoresearch.py history     # Full history
python3 autoresearch.py baseline    # Run + log baseline

# Run with custom params
python3 evaluate.py --params suggested_params.json --n-test 50
```

## Rules
- WORK ON `dev` branch -- never commit directly to main
- CREATE PRs for improvements -- this is how progress is tracked on GitHub
- ALWAYS log experiments with `autoresearch.py log`
- ALWAYS use `autoresearch.py suggest` before choosing parameters
- ALWAYS push to GitHub after finding improvements
- NEVER modify evaluate.py (the scoring function)
- ngspice is at `/usr/local/bin/ngspice`
- Target: analog accuracy >= 80% on 50 test digits

## What to Optimize
1. **Analog parameters** (saved in suggested_params.json):
   - G_scale: conductance scaling -- too small = noise, too large = nonlinearity
   - diode_n: ReLU sharpness -- ideal=0.001, real silicon=1-2
   - R_tia: transimpedance gain
   - R_pulldown: ReLU pull-down
   - V_high: input voltage range
2. **Architecture** (requires retraining):
   - Hidden layer size (32, 64, 128)
   - Weight quantization / clipping
3. **Circuit techniques**:
   - Opamp-based ReLU instead of diode
   - Voltage clamping / scaling between layers
   - Bias compensation circuits

## GitHub Repo
https://github.com/edonD/analog-nn-inference
