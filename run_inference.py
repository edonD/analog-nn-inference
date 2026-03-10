#!/usr/bin/env python3
"""
Run neural network inference through BOTH the Python model and the ngspice
analog circuit, then compare results.

Usage:
    python run_inference.py                    # Run all test inputs
    python run_inference.py --input "hel"      # Single input
    python run_inference.py --digital-only     # Skip ngspice, just digital
    python run_inference.py --save             # Save results to inference_results.json
"""

import argparse
import json
import os
import subprocess
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Constants (must match train_model.py)
# ---------------------------------------------------------------------------
VOCAB = {" ": 0}
for _i, _ch in enumerate("abcdefghijklmnopqrstuvwxyz", start=1):
    VOCAB[_ch] = _i
INV_VOCAB = {v: k for k, v in VOCAB.items()}

VOCAB_SIZE = 27
CONTEXT_LEN = 3
INPUT_DIM = CONTEXT_LEN * VOCAB_SIZE   # 81
HIDDEN_DIM = 32
OUTPUT_DIM = VOCAB_SIZE                 # 27

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_INPUTS = [
    "hel", "wor", "the", "and", "com", "net", "cir",
    "vol", "res", "tra", "pow", "sig", "fre", "ana",
    "dig", "chi", "des", "loo", "neu", "pro",
]


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def load_weights(weights_dir=None):
    """Load weights from weights.json and return numpy arrays + raw dict."""
    if weights_dir is None:
        weights_dir = SCRIPT_DIR
    path = os.path.join(weights_dir, "weights.json")
    with open(path, "r") as f:
        data = json.load(f)
    W1 = np.array(data["W1"])
    b1 = np.array(data["b1"])
    W2 = np.array(data["W2"])
    b2 = np.array(data["b2"])
    return W1, b1, W2, b2, data


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------
def encode_input(input_text):
    """Encode a 3-character string into an 81-dimensional one-hot vector."""
    text = input_text.lower()
    text = "".join(ch if ch in VOCAB else " " for ch in text)
    if len(text) < CONTEXT_LEN:
        text = " " * (CONTEXT_LEN - len(text)) + text
    ctx = text[-CONTEXT_LEN:]

    x = np.zeros(INPUT_DIM, dtype=np.float64)
    for pos in range(CONTEXT_LEN):
        x[pos * VOCAB_SIZE + VOCAB[ctx[pos]]] = 1.0
    return x


def char_label(idx):
    """Return a display-friendly character label for a vocab index."""
    ch = INV_VOCAB[idx]
    return repr(ch) if ch == " " else ch


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def softmax(logits):
    """Numerically stable softmax for a 1-D vector."""
    shifted = logits - logits.max()
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum()


# ---------------------------------------------------------------------------
# Digital inference (numpy forward pass)
# ---------------------------------------------------------------------------
def digital_inference(input_text, W1, b1, W2, b2):
    """
    Run the trained model in numpy.

    Returns:
        probs:       (27,) softmax probability vector
        pred_idx:    int, argmax index
        hidden:      (32,) hidden layer activations (post-ReLU)
        input_vec:   (81,) one-hot input vector
    """
    x = encode_input(input_text)
    z1 = x @ W1 + b1
    hidden = relu(z1)
    logits = hidden @ W2 + b2
    probs = softmax(logits)
    pred_idx = int(np.argmax(probs))
    return probs, pred_idx, hidden, x


# ---------------------------------------------------------------------------
# Analog inference (ngspice)
# ---------------------------------------------------------------------------
def analog_inference(input_text, W1, b1, W2, b2):
    """
    Run inference through the ngspice analog circuit.

    Steps:
        1. Call generate_netlist.py to create the ngspice circuit
        2. Run ngspice in batch mode
        3. Parse output voltages from nn_output.txt

    Returns:
        outputs:   (27,) array of output voltages (or None if ngspice unavailable)
        pred_idx:  int argmax index (or None)
        error_msg: str if something went wrong, else None
    """
    netlist_script = os.path.join(SCRIPT_DIR, "generate_netlist.py")
    circuit_file = os.path.join(SCRIPT_DIR, "analog_nn.cir")
    output_file = os.path.join(SCRIPT_DIR, "nn_output.txt")

    # --- Step 1: Generate netlist ---
    if not os.path.isfile(netlist_script):
        return None, None, "generate_netlist.py not found"

    try:
        gen_result = subprocess.run(
            [sys.executable, netlist_script, "--input", input_text],
            capture_output=True, text=True, timeout=30, cwd=SCRIPT_DIR,
        )
        if gen_result.returncode != 0:
            return None, None, f"generate_netlist.py failed: {gen_result.stderr.strip()}"
    except FileNotFoundError:
        return None, None, "Python interpreter not found"
    except subprocess.TimeoutExpired:
        return None, None, "generate_netlist.py timed out"

    # --- Step 2: Run ngspice ---
    if not os.path.isfile(circuit_file):
        return None, None, f"Circuit file not generated: {circuit_file}"

    # Try several common ngspice locations
    ngspice_cmd = None
    for candidate in ["ngspice", "/usr/local/bin/ngspice", "/usr/bin/ngspice"]:
        try:
            subprocess.run(
                [candidate, "--version"],
                capture_output=True, timeout=5,
            )
            ngspice_cmd = candidate
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if ngspice_cmd is None:
        return None, None, "ngspice not found on this system"

    try:
        sim_result = subprocess.run(
            [ngspice_cmd, "-b", circuit_file],
            capture_output=True, text=True, timeout=120, cwd=SCRIPT_DIR,
        )
    except subprocess.TimeoutExpired:
        return None, None, "ngspice simulation timed out"

    # --- Step 3: Parse output ---
    outputs = parse_ngspice_output(output_file)
    if outputs is None:
        # Try parsing from stdout as fallback
        outputs = parse_ngspice_stdout(sim_result.stdout)

    if outputs is None:
        return None, None, "Could not parse ngspice output"

    pred_idx = int(np.argmax(outputs))
    return outputs, pred_idx, None


def parse_ngspice_output(filepath):
    """
    Parse nn_output.txt for output node voltages.

    wrdata format from .op: alternating (0.0, value) pairs on one line.
    """
    if not os.path.isfile(filepath):
        return None

    try:
        with open(filepath, "r") as f:
            content = f.read().strip()
    except IOError:
        return None

    if not content:
        return None

    # Parse all numbers from the file
    tokens = content.split()
    all_nums = []
    for t in tokens:
        try:
            all_nums.append(float(t))
        except ValueError:
            continue

    # wrdata from .op produces pairs: (0.0, value) for each saved variable
    # Extract every other value (the actual voltages, skip the time=0 entries)
    if len(all_nums) >= OUTPUT_DIM * 2:
        values = [all_nums[i * 2 + 1] for i in range(min(OUTPUT_DIM, len(all_nums) // 2))]
        if len(values) == OUTPUT_DIM:
            return np.array(values)

    # Fallback: try to use all values directly
    if len(all_nums) >= OUTPUT_DIM:
        return np.array(all_nums[:OUTPUT_DIM])

    return None


def parse_ngspice_stdout(stdout_text):
    """Fallback: extract output voltages from ngspice stdout print statements.

    Handles format like: out_0 (' ') = -18.3159
    """
    if not stdout_text:
        return None

    import re
    # Match lines like: out_0 (' ') = -18.3159 or out_0 = 1.234
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

    values = []
    for line in stdout_text.split("\n"):
        line = line.strip()
        # Look for lines like "out0 = 1.234e-01" or "v(out0) = ..."
        if "out" in line.lower() and "=" in line:
            parts = line.split("=")
            try:
                val = float(parts[-1].strip())
                values.append(val)
            except ValueError:
                continue

    if len(values) >= OUTPUT_DIM:
        return np.array(values[:OUTPUT_DIM])
    return None


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def max_abs_error(a, b):
    """Compute maximum absolute error between two vectors."""
    return float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------
def format_top3(values, is_voltage=False):
    """Format top-3 predictions as a compact string."""
    top_indices = np.argsort(values)[::-1][:3]
    parts = []
    for idx in top_indices:
        ch = char_label(idx)
        val = values[idx]
        if is_voltage:
            parts.append(f"{ch}({val:.2f})")
        else:
            parts.append(f"{ch}({val:.2f})")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main inference runner
# ---------------------------------------------------------------------------
def run_all_inferences(test_inputs, digital_only=False, verbose=True):
    """
    Run inference on all test inputs and return structured results.

    Returns a list of dicts, one per test input.
    """
    W1, b1, W2, b2, weights_data = load_weights()

    results = []

    if verbose:
        print()
        print("=" * 70)
        print("  Analog Neural Network Inference")
        print("=" * 70)
        print()

    for input_text in test_inputs:
        result = {"input": input_text}

        # --- Digital inference ---
        d_probs, d_pred, d_hidden, d_input = digital_inference(
            input_text, W1, b1, W2, b2,
        )
        result["digital_probs"] = d_probs.tolist()
        result["digital_pred_idx"] = d_pred
        result["digital_pred_char"] = char_label(d_pred)
        result["digital_pred_prob"] = float(d_probs[d_pred])
        result["digital_hidden"] = d_hidden.tolist()
        result["digital_input"] = d_input.tolist()

        # --- Analog inference ---
        a_outputs = None
        a_pred = None
        a_error_msg = None
        if not digital_only:
            a_outputs, a_pred, a_error_msg = analog_inference(
                input_text, W1, b1, W2, b2,
            )

        if a_outputs is not None:
            result["analog_outputs"] = a_outputs.tolist()
            result["analog_pred_idx"] = a_pred
            result["analog_pred_char"] = char_label(a_pred)
            result["analog_pred_volt"] = float(a_outputs[a_pred])
            result["match"] = d_pred == a_pred
            result["cosine_sim"] = cosine_similarity(d_probs, a_outputs)
            result["max_error"] = max_abs_error(d_probs, a_outputs)
            result["analog_available"] = True
        else:
            result["analog_available"] = False
            result["analog_error"] = a_error_msg
            result["match"] = None
            result["cosine_sim"] = None
            result["max_error"] = None

        results.append(result)

        # --- Print result ---
        if verbose:
            print(f'Input: "{input_text}"')

            d_top3 = format_top3(d_probs, is_voltage=False)
            print(
                f"  Digital prediction:  "
                f"'{result['digital_pred_char']}' "
                f"(prob: {result['digital_pred_prob']:.2f})  "
                f"top-3: {d_top3}"
            )

            if result["analog_available"]:
                a_top3 = format_top3(a_outputs, is_voltage=True)
                match_str = "YES" if result["match"] else "NO"
                print(
                    f"  Analog prediction:   "
                    f"'{result['analog_pred_char']}' "
                    f"(volt: {result['analog_pred_volt']:.2f})  "
                    f"top-3: {a_top3}"
                )
                print(
                    f"  Match: {match_str} | "
                    f"Cosine sim: {result['cosine_sim']:.3f} | "
                    f"Max error: {result['max_error']:.2f}V"
                )
            elif digital_only:
                print("  Analog prediction:   [skipped -- digital-only mode]")
            else:
                print(
                    f"  Analog prediction:   [unavailable: {result['analog_error']}]"
                )

            print()

    # --- Summary ---
    if verbose:
        print("=" * 70)
        print("  Summary")
        print("=" * 70)
        print(f"  Total:    {len(results)} test cases")

        analog_results = [r for r in results if r["analog_available"]]
        if analog_results:
            matches = sum(1 for r in analog_results if r["match"])
            total_analog = len(analog_results)
            pct = 100.0 * matches / total_analog if total_analog > 0 else 0.0
            avg_cos = np.mean([r["cosine_sim"] for r in analog_results])
            avg_err = np.mean([r["max_error"] for r in analog_results])
            print(f"  Matches:  {matches}/{total_analog} ({pct:.1f}%)")
            print(f"  Avg cosine similarity: {avg_cos:.3f}")
            print(f"  Avg max error: {avg_err:.3f}V")
        elif digital_only:
            # Show digital-only summary
            print("  Mode:     digital-only (analog inference skipped)")
            # Show accuracy breakdown: how many unique predictions
            preds = [r["digital_pred_char"] for r in results]
            unique_preds = set(preds)
            print(f"  Unique predicted chars: {len(unique_preds)} -- {sorted(unique_preds)}")
        else:
            print("  Analog:   not available (ngspice / generate_netlist.py missing)")

        # Digital prediction summary table
        print()
        print("  Input  | Digital Pred | Prob  ", end="")
        if analog_results:
            print("| Analog Pred | Volt  | Match | CosSim | MaxErr")
        else:
            print()
        print("  " + "-" * (66 if analog_results else 34))

        for r in results:
            line = (
                f"  {r['input']:<5s}  | "
                f"{r['digital_pred_char']:>11s} | "
                f"{r['digital_pred_prob']:.3f} "
            )
            if r["analog_available"]:
                match_str = " YES " if r["match"] else " NO  "
                line += (
                    f"| {r['analog_pred_char']:>11s} | "
                    f"{r['analog_pred_volt']:.3f} | "
                    f"{match_str} | "
                    f"{r['cosine_sim']:.3f}  | "
                    f"{r['max_error']:.3f}"
                )
            line = line.rstrip()
            print(line)

        print("=" * 70)
        print()

    return results


# ---------------------------------------------------------------------------
# Save / load results
# ---------------------------------------------------------------------------
def save_results(results, filepath=None):
    """Save inference results to JSON for the dashboard."""
    if filepath is None:
        filepath = os.path.join(SCRIPT_DIR, "inference_results.json")

    # Also include weight metadata for the dashboard
    _, _, _, _, weights_data = load_weights()

    output = {
        "results": results,
        "weights": {
            "W1": weights_data["W1"],
            "b1": weights_data["b1"],
            "W2": weights_data["W2"],
            "b2": weights_data["b2"],
            "architecture": weights_data["architecture"],
            "vocab": weights_data["vocab"],
            "training_accuracy": weights_data.get("training_accuracy"),
            "training_loss": weights_data.get("training_loss"),
        },
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {filepath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run neural network inference (digital + analog) and compare results.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help='Single 3-character input to test (e.g. "hel")',
    )
    parser.add_argument(
        "--digital-only",
        action="store_true",
        help="Skip ngspice analog inference, run digital only",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to inference_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for output JSON (default: inference_results.json in script dir)",
    )
    args = parser.parse_args()

    if args.input:
        test_inputs = [args.input]
    else:
        test_inputs = TEST_INPUTS

    results = run_all_inferences(
        test_inputs,
        digital_only=args.digital_only,
        verbose=True,
    )

    if args.save or args.output:
        save_results(results, filepath=args.output)


if __name__ == "__main__":
    main()
