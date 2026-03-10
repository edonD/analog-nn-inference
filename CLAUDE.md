# Analog MNIST Autoresearch Project

An analog circuit that classifies handwritten digits using resistive crossbar arrays.
14x14 MNIST, realistic silicon diodes, 5% resistor mismatch. Pure physics inference.

Read `program.md` for full instructions. This is your bible.

## Quick Start
1. Read `program.md` completely
2. Run `python3 train_model.py --mode 14x14 --hidden 64 --epochs 2000` to train
3. Run `python3 evaluate.py --n-test 50` for baseline score
4. Run `python3 autoresearch.py baseline` to log baseline
5. Set up branches: `git checkout dev 2>/dev/null || git checkout -b dev`
6. Start the experiment loop (see program.md)

## Key Commands
```bash
# Train digital model (14x14 MNIST, 196->64->10)
python3 train_model.py --mode 14x14 --hidden 64 --epochs 2000

# Generate circuit for a single digit
python3 generate_circuit.py --digit-index 0

# Run evaluation (score the analog design)
python3 evaluate.py --n-test 50 --mode crossbar 2>&1 | tee run.log

# Autoresearcher commands
python3 autoresearch.py status
python3 autoresearch.py suggest
python3 autoresearch.py log --notes "description"
python3 autoresearch.py history

# Update GitHub results page
python3 update_results.py
```

## Rules
- WORK ON `dev` branch -- never commit directly to main
- CREATE PRs for improvements -- this is how progress is tracked on GitHub
- ALWAYS push to GitHub after finding improvements
- ALWAYS run `python3 update_results.py` before committing improvements
- ALWAYS log experiments with `autoresearch.py log`
- NEVER modify evaluate.py (the scoring function)
- ngspice is at `/usr/local/bin/ngspice`
- Target: analog accuracy >= 80% on 50 test digits

## PR Workflow (MUST follow for every improvement)
```bash
# After finding a better score:
python3 update_results.py
git add -A
git commit -m "exp: score X.XXX -- description"
git push origin dev

gh pr create --base main --head dev \
  --title "[MNIST] score: X.XX -- description" \
  --body "## Metrics
- Analog accuracy: XX.X%
- Match rate: XX.X%
- Total score: X.XXXX
## Changes
- what was changed"

gh pr merge --merge --delete-branch=false
git checkout main && git pull && git checkout dev && git merge main
```

## What to Optimize
The hard constraints make this challenging:
- **Realistic diode** (n=1.0): ~0.7V forward drop kills ReLU precision
- **5% resistor mismatch**: random variation on every resistor
- **14x14 input** (196 pixels): 26,000+ resistors, more noise accumulation

Tunable parameters (saved in suggested_params.json):
1. G_scale: conductance scaling (critical for SNR)
2. diode_n: ReLU sharpness (1.0 = real, 0.001 = ideal)
3. diode_is: saturation current (affects turn-on voltage)
4. R_tia: transimpedance gain
5. R_pulldown: ReLU pull-down
6. V_high: input voltage range
7. mismatch_pct: resistor variation (0-10%)
8. mismatch_seed: reproducible randomness

Architecture changes (require retraining):
- Hidden layer size: 32, 64, 128
- Weight clipping / quantization-aware training

Circuit improvements (modify generate_circuit.py):
- Opamp-based ReLU instead of diode
- Voltage clamping between layers
- Bias compensation circuits

## GitHub Repo
https://github.com/edonD/analog-nn-inference
