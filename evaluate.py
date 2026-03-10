#!/usr/bin/env python3
"""
Evaluate analog MNIST classifier accuracy.

Runs multiple test digits through ngspice and compares analog predictions
to digital (ground truth) predictions.

Scoring:
    primary_score   = analog_accuracy (% of test digits correctly classified)
    match_score     = analog_vs_digital_match (% matching digital predictions)
    fidelity_score  = avg cosine similarity between analog and digital output vectors

Usage:
    python evaluate.py                    # Evaluate 50 test digits
    python evaluate.py --n-test 100       # More test digits
    python evaluate.py --mode hybrid      # Use behavioral matmul instead of crossbar
    python evaluate.py --params params.json  # Custom analog parameters
    python evaluate.py --timeout 300      # Longer timeout per simulation
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIM = 10


def load_weights(weights_path=None):
    if weights_path is None:
        weights_path = SCRIPT_DIR / "weights.json"
    with open(weights_path, "r") as f:
        data = json.load(f)
    return {
        "W1": np.array(data["W1"]),
        "b1": np.array(data["b1"]),
        "W2": np.array(data["W2"]),
        "b2": np.array(data["b2"]),
        "architecture": data.get("architecture", {}),
    }


def digital_inference(x, weights):
    """Run numpy forward pass, return logits and prediction."""
    z1 = x @ weights["W1"] + weights["b1"]
    a1 = np.maximum(0, z1)
    logits = a1 @ weights["W2"] + weights["b2"]
    pred = int(np.argmax(logits))
    return logits, pred


def parse_ngspice_output(filepath):
    """Parse analog_output.txt for output voltages."""
    if not os.path.isfile(filepath):
        return None
    with open(filepath, "r") as f:
        content = f.read().strip()
    if not content:
        return None

    tokens = content.split()
    nums = []
    for t in tokens:
        try:
            nums.append(float(t))
        except ValueError:
            continue

    # wrdata from .op: alternating (0.0, value) pairs
    if len(nums) >= OUTPUT_DIM * 2:
        values = [nums[i * 2 + 1] for i in range(min(OUTPUT_DIM, len(nums) // 2))]
        if len(values) == OUTPUT_DIM:
            return np.array(values)

    if len(nums) >= OUTPUT_DIM:
        return np.array(nums[:OUTPUT_DIM])
    return None


def parse_ngspice_stdout(stdout_text):
    """Parse output voltages from ngspice stdout."""
    if not stdout_text:
        return None

    pattern = re.compile(r"out_(\d+)\s*(?:\([^)]*\))?\s*=\s*([-\d.eE+]+)")
    values = {}
    for line in stdout_text.split("\n"):
        m = pattern.search(line)
        if m:
            idx = int(m.group(1))
            val = float(m.group(2))
            values[idx] = val

    if len(values) >= OUTPUT_DIM:
        return np.array([values.get(i, 0.0) for i in range(OUTPUT_DIM)])
    return None


def find_ngspice():
    """Find ngspice binary."""
    for candidate in ["ngspice", "/usr/local/bin/ngspice", "/usr/bin/ngspice"]:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
            return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def run_single_eval(pixel_values, true_label, weights, params_path, mode, ngspice_cmd, timeout):
    """
    Run a single digit through the analog circuit and return results.

    Returns dict with: true_label, digital_pred, analog_pred, analog_outputs, match, correct, etc.
    """
    result = {
        "true_label": int(true_label),
        "analog_available": False,
    }

    # Digital inference
    d_logits, d_pred = digital_inference(pixel_values, weights)
    result["digital_pred"] = d_pred
    result["digital_logits"] = d_logits.tolist()
    result["digital_correct"] = (d_pred == true_label)

    # Generate circuit
    gen_cmd = [
        sys.executable, str(SCRIPT_DIR / "generate_circuit.py"),
        "--pixel-values", ",".join(f"{v:.6f}" for v in pixel_values),
        "--label", str(true_label),
        "--mode", mode,
    ]
    if params_path:
        gen_cmd.extend(["--params", params_path])

    try:
        gen_result = subprocess.run(gen_cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPT_DIR))
        if gen_result.returncode != 0:
            result["error"] = f"generate_circuit.py failed: {gen_result.stderr.strip()[:200]}"
            return result
    except subprocess.TimeoutExpired:
        result["error"] = "Circuit generation timed out"
        return result

    # Run ngspice
    circuit_file = str(SCRIPT_DIR / "analog_mnist.cir")
    output_file = str(SCRIPT_DIR / "analog_output.txt")

    # Remove old output file
    if os.path.exists(output_file):
        os.remove(output_file)

    try:
        t0 = time.time()
        sim_result = subprocess.run(
            [ngspice_cmd, "-b", circuit_file],
            capture_output=True, text=True, timeout=timeout, cwd=str(SCRIPT_DIR),
        )
        sim_time = time.time() - t0
        result["sim_time"] = sim_time
    except subprocess.TimeoutExpired:
        result["error"] = f"ngspice timed out after {timeout}s"
        return result

    # Parse output
    outputs = parse_ngspice_output(output_file)
    if outputs is None:
        outputs = parse_ngspice_stdout(sim_result.stdout)

    if outputs is None:
        result["error"] = "Could not parse ngspice output"
        return result

    a_pred = int(np.argmax(outputs))
    result["analog_available"] = True
    result["analog_pred"] = a_pred
    result["analog_outputs"] = outputs.tolist()
    result["analog_correct"] = (a_pred == true_label)
    result["match"] = (a_pred == d_pred)

    # Fidelity metrics
    norm_d = np.linalg.norm(d_logits)
    norm_a = np.linalg.norm(outputs)
    if norm_d > 1e-12 and norm_a > 1e-12:
        result["cosine_sim"] = float(np.dot(d_logits, outputs) / (norm_d * norm_a))
    else:
        result["cosine_sim"] = 0.0

    return result


def evaluate(n_test=50, mode="crossbar", params_path=None, weights_path=None,
             timeout=120, verbose=True):
    """Run full evaluation and return score dict."""

    # Load data
    sys.path.insert(0, str(SCRIPT_DIR))
    from train_model import load_dataset, train_test_split
    weights = load_weights(weights_path)
    img_size = weights.get("architecture", {}).get("img_size", 8) if hasattr(weights, 'get') else 8

    # Detect mode from weights
    n_in = weights["W1"].shape[0]
    ds_mode = "8x8" if n_in == 64 else "14x14"
    X, y, _ = load_dataset(ds_mode)
    _, _, X_test, y_test = train_test_split(X, y)
    ngspice_cmd = find_ngspice()
    if ngspice_cmd is None:
        print("ERROR: ngspice not found")
        sys.exit(1)

    n = min(n_test, len(X_test))
    results = []

    if verbose:
        print(f"\nEvaluating {n} test digits (mode={mode})...")
        print(f"ngspice: {ngspice_cmd}")
        print("-" * 60)

    for idx in range(n):
        r = run_single_eval(
            X_test[idx], int(y_test[idx]), weights, params_path, mode, ngspice_cmd, timeout
        )
        r["index"] = idx
        results.append(r)

        if verbose:
            status = ""
            if r["analog_available"]:
                sym = "OK" if r["analog_correct"] else "WRONG"
                match = "match" if r["match"] else "mismatch"
                status = f"true={r['true_label']} analog={r['analog_pred']} digital={r['digital_pred']} [{sym}] [{match}] {r.get('sim_time', 0):.1f}s"
            else:
                status = f"true={r['true_label']} ERROR: {r.get('error', 'unknown')}"
            print(f"  [{idx+1:3d}/{n}] {status}")

    # Compute scores
    analog_results = [r for r in results if r["analog_available"]]
    n_analog = len(analog_results)

    if n_analog == 0:
        scores = {
            "analog_accuracy": 0.0,
            "digital_accuracy": 0.0,
            "match_rate": 0.0,
            "avg_cosine_sim": 0.0,
            "total_score": 0.0,
            "n_test": n,
            "n_analog_ok": 0,
            "n_errors": n,
        }
    else:
        analog_correct = sum(1 for r in analog_results if r["analog_correct"])
        digital_correct = sum(1 for r in results if r["digital_correct"])
        matches = sum(1 for r in analog_results if r["match"])
        avg_cos = np.mean([r["cosine_sim"] for r in analog_results])
        avg_time = np.mean([r.get("sim_time", 0) for r in analog_results])

        analog_acc = analog_correct / n_analog
        digital_acc = digital_correct / n
        match_rate = matches / n_analog

        # Total score: weighted combination
        # 60% analog accuracy, 30% match rate, 10% fidelity
        total_score = 0.6 * analog_acc + 0.3 * match_rate + 0.1 * max(0, avg_cos)

        scores = {
            "analog_accuracy": round(analog_acc, 4),
            "digital_accuracy": round(digital_acc, 4),
            "match_rate": round(match_rate, 4),
            "avg_cosine_sim": round(float(avg_cos), 4),
            "total_score": round(total_score, 4),
            "n_test": n,
            "n_analog_ok": n_analog,
            "n_errors": n - n_analog,
            "analog_correct": analog_correct,
            "digital_correct": digital_correct,
            "matches": matches,
            "avg_sim_time": round(avg_time, 2),
        }

    if verbose:
        print()
        print("=" * 60)
        print("  EVALUATION RESULTS")
        print("=" * 60)
        print(f"  Test digits:         {n}")
        print(f"  Simulated OK:        {scores['n_analog_ok']}")
        print(f"  Simulation errors:   {scores['n_errors']}")
        print(f"  Digital accuracy:    {scores['digital_accuracy']:.1%}")
        print(f"  Analog accuracy:     {scores['analog_accuracy']:.1%}")
        print(f"  Analog-digital match:{scores['match_rate']:.1%}")
        print(f"  Avg cosine sim:      {scores['avg_cosine_sim']:.4f}")
        print(f"  Avg sim time:        {scores.get('avg_sim_time', 0):.1f}s")
        print(f"  ----------------------------------------")
        print(f"  TOTAL SCORE:         {scores['total_score']:.4f}")
        print("=" * 60)

        # Confusion matrix
        if analog_results:
            print("\n  Analog confusion matrix (rows=true, cols=predicted):")
            cm = np.zeros((10, 10), dtype=int)
            for r in analog_results:
                cm[r["true_label"]][r["analog_pred"]] += 1
            print("       " + "".join(f"{i:>4d}" for i in range(10)))
            for i in range(10):
                row = "".join(f"{cm[i][j]:>4d}" for j in range(10))
                print(f"    {i}: {row}")

    # Save results
    output = {
        "scores": scores,
        "results": results,
        "mode": mode,
        "params_path": params_path,
    }
    results_path = SCRIPT_DIR / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    if verbose:
        print(f"\nResults saved to {results_path}")

    return scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate analog MNIST classifier.")
    parser.add_argument("--n-test", type=int, default=50, help="Number of test digits (default: 50)")
    parser.add_argument("--mode", choices=["crossbar", "hybrid"], default="crossbar")
    parser.add_argument("--params", type=str, default=None, help="Analog params JSON")
    parser.add_argument("--weights", type=str, default=None, help="Weights JSON")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per sim (seconds)")
    args = parser.parse_args()

    scores = evaluate(
        n_test=args.n_test,
        mode=args.mode,
        params_path=args.params,
        weights_path=args.weights,
        timeout=args.timeout,
    )

    # Exit with error if score is 0
    sys.exit(0 if scores["total_score"] > 0 else 1)


if __name__ == "__main__":
    main()
