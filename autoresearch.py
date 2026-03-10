#!/usr/bin/env python3
"""
Autoresearcher for analog MNIST classifier optimization.

Manages experiment history, suggests parameter changes, and tracks
the optimization journey from initial analog accuracy to maximized score.

Hard constraints (NOT tunable - realistic silicon):
    diode_n       = 1.0    (real silicon diode ideality)
    mismatch_pct  = 5.0    (5% manufacturing resistor variation)

The autoresearcher tunes these circuit design parameters:
    G_scale     - Conductance scaling (weight-to-resistance mapping)
    diode_is    - ReLU diode saturation current
    R_tia       - Transimpedance amplifier gain
    R_pulldown  - ReLU pull-down resistance
    V_high      - Input voltage scale
    reltol      - ngspice convergence tolerance

Usage:
    python autoresearch.py status                # Show current best & history
    python autoresearch.py suggest               # Get next experiment suggestion
    python autoresearch.py log --score 0.65 --params '{"G_scale": 2e-3}'
    python autoresearch.py history               # Full experiment history
    python autoresearch.py baseline              # Run baseline evaluation
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
HISTORY_FILE = SCRIPT_DIR / "experiment_history.json"

# ---------------------------------------------------------------------------
# Hard constraints (realistic silicon — NOT tunable)
# ---------------------------------------------------------------------------
FIXED_PARAMS = {
    "diode_n": 1.0,          # Real silicon diode ideality factor
    "mismatch_pct": 5.0,     # 5% manufacturing resistor variation
    "mismatch_seed": 42,     # Reproducible randomness
}

# ---------------------------------------------------------------------------
# Tunable parameter space (circuit design knobs)
# ---------------------------------------------------------------------------
PARAM_SPACE = {
    "G_scale": {
        "type": "log",
        "min": 1e-5,
        "max": 1e-1,
        "default": 1e-3,
        "description": "Conductance scale factor (weight=1 -> G=G_scale)",
    },
    "diode_is": {
        "type": "log",
        "min": 1e-18,
        "max": 1e-10,
        "default": 1e-14,
        "description": "Diode saturation current",
    },
    "R_tia": {
        "type": "log",
        "min": 100,
        "max": 100000,
        "default": 1000,
        "description": "Transimpedance feedback resistance",
    },
    "R_pulldown": {
        "type": "log",
        "min": 10000,
        "max": 100000000,
        "default": 1000000,
        "description": "ReLU pull-down resistance",
    },
    "V_high": {
        "type": "linear",
        "min": 0.1,
        "max": 3.3,
        "default": 1.0,
        "description": "Input voltage for pixel=1.0",
    },
    "reltol": {
        "type": "log",
        "min": 1e-6,
        "max": 1e-2,
        "default": 1e-4,
        "description": "ngspice relative tolerance",
    },
}


def build_full_params(tunable_params):
    """Combine tunable params with fixed hard-mode constraints."""
    full = dict(FIXED_PARAMS)
    full.update(tunable_params)
    return full


def load_history():
    """Load experiment history."""
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return {"experiments": [], "best_score": 0.0, "best_exp_id": None}


def save_history(history):
    """Save experiment history."""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def log_experiment(params, scores, notes=""):
    """Log an experiment result."""
    history = load_history()
    exp_id = len(history["experiments"]) + 1
    entry = {
        "id": exp_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "scores": scores,
        "notes": notes,
    }
    history["experiments"].append(entry)

    total_score = scores.get("total_score", 0)
    if total_score > history["best_score"]:
        history["best_score"] = total_score
        history["best_exp_id"] = exp_id
        entry["is_best"] = True
        print(f"  NEW BEST SCORE: {total_score:.4f} (exp #{exp_id})")
    else:
        entry["is_best"] = False

    save_history(history)
    return exp_id


def get_status():
    """Print current optimization status."""
    history = load_history()
    n = len(history["experiments"])

    print("\n" + "=" * 50)
    print("  AUTORESEARCH STATUS")
    print("=" * 50)
    print(f"  Total experiments: {n}")
    print(f"  Best score:        {history['best_score']:.4f}")

    if n > 0:
        best_id = history["best_exp_id"]
        best_exp = next((e for e in history["experiments"] if e["id"] == best_id), None)
        if best_exp:
            print(f"  Best experiment:   #{best_id}")
            print(f"  Best params:")
            for k, v in best_exp["params"].items():
                print(f"    {k}: {v}")
            print(f"  Best scores:")
            for k, v in best_exp["scores"].items():
                print(f"    {k}: {v}")

        # Recent experiments
        print(f"\n  Recent experiments:")
        for exp in history["experiments"][-5:]:
            score = exp["scores"].get("total_score", 0)
            best_marker = " *BEST*" if exp.get("is_best") else ""
            print(f"    #{exp['id']:3d} | score={score:.4f} | {exp['timestamp']}{best_marker}")

    print("=" * 50)


def suggest_next():
    """Suggest next experiment parameters based on history."""
    import numpy as np
    history = load_history()
    experiments = history["experiments"]
    n = len(experiments)

    if n == 0:
        # First experiment: use defaults
        print("\n  Suggestion: Run baseline with default parameters")
        params = build_full_params({k: v["default"] for k, v in PARAM_SPACE.items()})
        print(f"  Params: {json.dumps(params, indent=4)}")
        return params

    # Get best experiment params as starting point
    best_id = history["best_exp_id"]
    best_exp = next((e for e in experiments if e["id"] == best_id), experiments[-1])
    best_params = best_exp["params"]

    # Strategy selection based on experiment count
    rng = np.random.RandomState(n + 42)

    if n < 5:
        # Early phase: explore parameter space broadly
        strategy = "latin_hypercube"
    elif n < 15:
        # Mid phase: perturb best params
        strategy = "perturbation"
    else:
        # Late phase: fine-tune best params
        strategy = "fine_tune"

    print(f"\n  Strategy: {strategy} (experiment #{n+1})")

    if strategy == "latin_hypercube":
        # Sample random point in parameter space
        params = {}
        for name, spec in PARAM_SPACE.items():
            if spec["type"] == "log":
                log_min = np.log10(spec["min"])
                log_max = np.log10(spec["max"])
                val = 10 ** rng.uniform(log_min, log_max)
            else:
                val = rng.uniform(spec["min"], spec["max"])
            params[name] = float(val)

    elif strategy == "perturbation":
        # Perturb best params by 10-50%
        params = {}
        for name, spec in PARAM_SPACE.items():
            base = best_params.get(name, spec["default"])
            if spec["type"] == "log":
                log_val = np.log10(base)
                log_range = np.log10(spec["max"]) - np.log10(spec["min"])
                perturbation = rng.normal(0, log_range * 0.1)
                val = 10 ** np.clip(log_val + perturbation,
                                     np.log10(spec["min"]), np.log10(spec["max"]))
            else:
                range_size = spec["max"] - spec["min"]
                perturbation = rng.normal(0, range_size * 0.1)
                val = np.clip(base + perturbation, spec["min"], spec["max"])
            params[name] = float(val)

    else:  # fine_tune
        # Small perturbation (2-5%) around best
        params = {}
        for name, spec in PARAM_SPACE.items():
            base = best_params.get(name, spec["default"])
            if spec["type"] == "log":
                log_val = np.log10(base)
                log_range = np.log10(spec["max"]) - np.log10(spec["min"])
                perturbation = rng.normal(0, log_range * 0.03)
                val = 10 ** np.clip(log_val + perturbation,
                                     np.log10(spec["min"]), np.log10(spec["max"]))
            else:
                range_size = spec["max"] - spec["min"]
                perturbation = rng.normal(0, range_size * 0.03)
                val = np.clip(base + perturbation, spec["min"], spec["max"])
            params[name] = float(val)

    print(f"  Suggested params:")
    for k, v in params.items():
        diff = ""
        if k in best_params:
            ratio = v / best_params[k] if best_params[k] != 0 else 0
            diff = f" (vs best: {ratio:.2f}x)"
        print(f"    {k}: {v:.6g}{diff}")

    # Add fixed hard-mode constraints
    full_params = build_full_params(params)
    print(f"  Fixed constraints: diode_n={FIXED_PARAMS['diode_n']}, "
          f"mismatch_pct={FIXED_PARAMS['mismatch_pct']}%")

    # Save suggestion
    params_path = SCRIPT_DIR / "suggested_params.json"
    with open(params_path, "w") as f:
        json.dump(full_params, f, indent=2)
    print(f"\n  Saved to {params_path}")
    print(f"  Run: python evaluate.py --params {params_path}")

    return full_params


def show_history():
    """Show full experiment history."""
    history = load_history()
    if not history["experiments"]:
        print("No experiments yet.")
        return

    print(f"\n{'='*80}")
    print(f"  EXPERIMENT HISTORY ({len(history['experiments'])} experiments)")
    print(f"{'='*80}")
    print(f"  {'#':>4s} | {'Score':>7s} | {'Analog Acc':>10s} | {'Match':>7s} | {'CosSim':>7s} | {'Time':>19s}")
    print(f"  {'-'*70}")

    for exp in history["experiments"]:
        s = exp["scores"]
        marker = " *" if exp.get("is_best") else ""
        print(f"  {exp['id']:4d} | "
              f"{s.get('total_score', 0):7.4f} | "
              f"{s.get('analog_accuracy', 0):10.4f} | "
              f"{s.get('match_rate', 0):7.4f} | "
              f"{s.get('avg_cosine_sim', 0):7.4f} | "
              f"{exp['timestamp']}{marker}")

    print(f"{'='*80}")


def run_baseline():
    """Run baseline evaluation with hard-mode default parameters."""
    import subprocess

    # Write hard-mode defaults to suggested_params.json
    default_tunable = {k: v["default"] for k, v in PARAM_SPACE.items()}
    full_params = build_full_params(default_tunable)
    params_path = SCRIPT_DIR / "suggested_params.json"
    with open(params_path, "w") as f:
        json.dump(full_params, f, indent=2)

    print(f"Running baseline with hard-mode constraints:")
    print(f"  diode_n={FIXED_PARAMS['diode_n']} (real silicon)")
    print(f"  mismatch_pct={FIXED_PARAMS['mismatch_pct']}%")
    print()

    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "evaluate.py"),
         "--n-test", "50", "--mode", "crossbar",
         "--params", str(params_path)],
        capture_output=True, text=True, cwd=str(SCRIPT_DIR),
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Parse and log results
    eval_path = SCRIPT_DIR / "eval_results.json"
    if eval_path.exists():
        with open(eval_path, "r") as f:
            data = json.load(f)
        scores = data["scores"]
        exp_id = log_experiment(full_params, scores, notes="baseline: hard mode (n=1.0, 5% mismatch)")
        print(f"\nLogged as experiment #{exp_id}")


def main():
    parser = argparse.ArgumentParser(description="Analog MNIST autoresearcher.")
    parser.add_argument("command", choices=["status", "suggest", "log", "history", "baseline"],
                        help="Command to run")
    parser.add_argument("--score", type=float, help="Total score (for log command)")
    parser.add_argument("--params", type=str, help="JSON params string (for log command)")
    parser.add_argument("--notes", type=str, default="", help="Notes (for log command)")
    args = parser.parse_args()

    if args.command == "status":
        get_status()
    elif args.command == "suggest":
        suggest_next()
    elif args.command == "log":
        if args.params is None:
            # Try to load from suggested_params.json
            sp = SCRIPT_DIR / "suggested_params.json"
            if sp.exists():
                with open(sp, "r") as f:
                    params = json.load(f)
            else:
                print("ERROR: --params required or suggested_params.json must exist")
                sys.exit(1)
        else:
            params = json.loads(args.params)

        # Try to load scores from eval_results.json
        eval_path = SCRIPT_DIR / "eval_results.json"
        if eval_path.exists():
            with open(eval_path, "r") as f:
                data = json.load(f)
            scores = data["scores"]
        elif args.score is not None:
            scores = {"total_score": args.score}
        else:
            print("ERROR: Run evaluate.py first or provide --score")
            sys.exit(1)

        exp_id = log_experiment(params, scores, args.notes)
        print(f"Logged experiment #{exp_id}")

    elif args.command == "history":
        show_history()
    elif args.command == "baseline":
        run_baseline()


if __name__ == "__main__":
    main()
