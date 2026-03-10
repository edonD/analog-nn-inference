#!/usr/bin/env python3
"""
Interactive demo: watch an analog circuit classify a handwritten digit.

Walks through every step so you can see exactly what happens:
  1. Display the 8x8 digit image
  2. Convert pixels to 64 input voltages
  3. Show the resistive crossbar circuit stats
  4. Run ngspice (pure analog physics)
  5. Read output voltages — one per digit class
  6. The highest voltage wins → that's the prediction
  7. Compare: analog circuit vs digital numpy

Usage:
    python demo.py                     # Run 5 random test digits
    python demo.py --n 10              # Run 10 digits
    python demo.py --digit-index 42    # Specific test digit
    python demo.py --all               # All test digits (full accuracy report)
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
INPUT_DIM = 64
HIDDEN_DIM = 32

# ANSI colors (works in most terminals)
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BAR = "\u2588"  # full block character


def load_data():
    """Load dataset and weights."""
    sys.path.insert(0, str(SCRIPT_DIR))
    from train_model import load_dataset, train_test_split
    X, y = load_dataset()
    _, _, X_test, y_test = train_test_split(X, y)

    with open(SCRIPT_DIR / "weights.json", "r") as f:
        data = json.load(f)
    W1 = np.array(data["W1"])
    b1 = np.array(data["b1"])
    W2 = np.array(data["W2"])
    b2 = np.array(data["b2"])

    return X_test, y_test, W1, b1, W2, b2


def digital_forward(x, W1, b1, W2, b2):
    """Numpy forward pass returning hidden activations and logits."""
    z1 = x @ W1 + b1
    h1 = np.maximum(0, z1)  # ReLU
    logits = h1 @ W2 + b2
    return h1, logits


def render_digit(pixels, label=None):
    """Render 8x8 digit as colored ASCII art."""
    img = pixels.reshape(8, 8)
    shades = " ░▒▓█"

    lines = []
    if label is not None:
        lines.append(f"  {BOLD}True digit: {label}{RESET}")
    lines.append(f"  {DIM}┌────────────────┐{RESET}")
    for row in img:
        line = ""
        for val in row:
            idx = min(int(val * (len(shades) - 1)), len(shades) - 1)
            if val > 0.5:
                line += f"{BOLD}{shades[idx]}{shades[idx]}{RESET}"
            elif val > 0.1:
                line += f"{DIM}{shades[idx]}{shades[idx]}{RESET}"
            else:
                line += "  "
        lines.append(f"  {DIM}│{RESET}{line}{DIM}│{RESET}")
    lines.append(f"  {DIM}└────────────────┘{RESET}")
    return "\n".join(lines)


def render_voltages_input(pixels):
    """Show which pixel voltages are being fed into the circuit."""
    img = pixels.reshape(8, 8)
    active = sum(1 for v in pixels if v > 0.05)
    lines = []
    lines.append(f"  {CYAN}64 voltage sources → one per pixel{RESET}")
    lines.append(f"  {DIM}Pixel intensity 0.0-1.0 maps to 0.0V-1.0V{RESET}")
    lines.append(f"  Active pixels (>0.05V): {BOLD}{active}/64{RESET}")
    lines.append("")

    # Show voltage grid
    lines.append(f"  {DIM}Input voltages (V):{RESET}")
    lines.append(f"  {DIM}┌" + "─" * 49 + f"┐{RESET}")
    for r in range(8):
        row_str = ""
        for c in range(8):
            v = img[r][c]
            if v > 0.5:
                row_str += f" {BOLD}{v:.2f}{RESET} "
            elif v > 0.1:
                row_str += f" {DIM}{v:.2f}{RESET} "
            else:
                row_str += f" {DIM} ·  {RESET} "
        lines.append(f"  {DIM}│{RESET}{row_str}{DIM}│{RESET}")
    lines.append(f"  {DIM}└" + "─" * 49 + f"┘{RESET}")
    return "\n".join(lines)


def render_circuit_info():
    """Show what the circuit looks like."""
    lines = []
    lines.append(f"  {MAGENTA}=== THE ANALOG CIRCUIT ==={RESET}")
    lines.append("")
    lines.append(f"  {DIM}64 pixel voltages (0-1V){RESET}")
    lines.append(f"  {DIM}        │{RESET}")
    lines.append(f"  {DIM}        ▼{RESET}")
    lines.append(f"  {BOLD}┌─────────────────────────────────┐{RESET}")
    lines.append(f"  {BOLD}│  LAYER 1: 64×32 Crossbar Array  │{RESET}")
    lines.append(f"  {BOLD}│  {CYAN}4,096 resistors{RESET}{BOLD} (differential)  │{RESET}")
    lines.append(f"  {BOLD}│                                 │{RESET}")
    lines.append(f"  {BOLD}│  {DIM}Ohm's law:  V/R = I  (multiply){RESET}{BOLD} │{RESET}")
    lines.append(f"  {BOLD}│  {DIM}KCL: ΣI = 0 at node  (sum){RESET}{BOLD}      │{RESET}")
    lines.append(f"  {BOLD}└────────────────┬────────────────┘{RESET}")
    lines.append(f"  {DIM}                 │{RESET}")
    lines.append(f"  {DIM}                 ▼{RESET}")
    lines.append(f"  {BOLD}┌─────────────────────────────────┐{RESET}")
    lines.append(f"  {BOLD}│  {YELLOW}32 Diode ReLU circuits{RESET}{BOLD}          │{RESET}")
    lines.append(f"  {BOLD}│  {DIM}pass positive, block negative{RESET}{BOLD}   │{RESET}")
    lines.append(f"  {BOLD}└────────────────┬────────────────┘{RESET}")
    lines.append(f"  {DIM}                 │{RESET}")
    lines.append(f"  {DIM}                 ▼{RESET}")
    lines.append(f"  {BOLD}┌─────────────────────────────────┐{RESET}")
    lines.append(f"  {BOLD}│  LAYER 2: 32×10 Crossbar Array  │{RESET}")
    lines.append(f"  {BOLD}│  {CYAN}640 resistors{RESET}{BOLD} (differential)    │{RESET}")
    lines.append(f"  {BOLD}└────────────────┬────────────────┘{RESET}")
    lines.append(f"  {DIM}                 │{RESET}")
    lines.append(f"  {DIM}                 ▼{RESET}")
    lines.append(f"  {BOLD}  10 output voltages (one per digit){RESET}")
    lines.append(f"  {DIM}  Highest voltage = predicted digit{RESET}")
    lines.append("")
    lines.append(f"  {DIM}Total: {BOLD}4,736 resistors{RESET}{DIM} + {BOLD}32 diodes{RESET}")
    lines.append(f"  {DIM}No code executes. Pure physics.{RESET}")
    return "\n".join(lines)


def render_output_bar(label, value, max_val, is_winner=False, is_true=False):
    """Render a horizontal bar for an output voltage."""
    bar_width = 40
    if max_val > 0:
        normalized = max(0, (value - min(0, max_val * -0.5)) / (max_val * 1.5 + 0.001))
    else:
        normalized = 0
    bar_len = int(normalized * bar_width)
    bar_len = max(0, min(bar_len, bar_width))

    if is_winner:
        color = GREEN
        marker = " ◀ PREDICTED"
    elif is_true and not is_winner:
        color = RED
        marker = " ◀ TRUE (missed)"
    else:
        color = DIM
        marker = ""

    bar = BAR * bar_len + "░" * (bar_width - bar_len)
    sign = "+" if value >= 0 else ""
    return f"  {color}digit {label}: {bar} {sign}{value:7.3f}V{marker}{RESET}"


def run_analog(pixels, true_label):
    """Generate circuit, run ngspice, parse results."""
    # Generate circuit
    pixel_str = ",".join(f"{v:.6f}" for v in pixels)
    gen_cmd = [
        sys.executable, str(SCRIPT_DIR / "generate_circuit.py"),
        "--pixel-values", pixel_str,
        "--label", str(true_label),
        "--mode", "crossbar",
    ]
    subprocess.run(gen_cmd, capture_output=True, text=True, timeout=30, cwd=str(SCRIPT_DIR))

    circuit_file = str(SCRIPT_DIR / "analog_mnist.cir")
    output_file = str(SCRIPT_DIR / "analog_output.txt")

    if os.path.exists(output_file):
        os.remove(output_file)

    # Find ngspice
    ngspice_cmd = None
    for candidate in ["ngspice", "/usr/local/bin/ngspice", "/usr/bin/ngspice"]:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, timeout=5)
            ngspice_cmd = candidate
            break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if ngspice_cmd is None:
        return None, "ngspice not found"

    # Run simulation
    t0 = time.time()
    result = subprocess.run(
        [ngspice_cmd, "-b", circuit_file],
        capture_output=True, text=True, timeout=120, cwd=str(SCRIPT_DIR),
    )
    sim_time = time.time() - t0

    # Parse output
    outputs = None
    if os.path.isfile(output_file):
        with open(output_file, "r") as f:
            content = f.read().strip()
        if content:
            tokens = content.split()
            nums = []
            for t in tokens:
                try:
                    nums.append(float(t))
                except ValueError:
                    continue
            if len(nums) >= OUTPUT_DIM * 2:
                outputs = np.array([nums[i * 2 + 1] for i in range(OUTPUT_DIM)])

    if outputs is None:
        # Try stdout
        pattern = re.compile(r"out_(\d+)\s*(?:\([^)]*\))?\s*=\s*([-\d.eE+]+)")
        values = {}
        for line in result.stdout.split("\n"):
            m = pattern.search(line)
            if m:
                values[int(m.group(1))] = float(m.group(2))
        if len(values) >= OUTPUT_DIM:
            outputs = np.array([values.get(i, 0.0) for i in range(OUTPUT_DIM)])

    if outputs is None:
        return None, "Could not parse output"

    return outputs, sim_time


def demo_single(idx, pixels, true_label, W1, b1, W2, b2, show_circuit=True):
    """Run full demo for a single digit."""

    print()
    print(f"{'='*60}")
    print(f"  {BOLD}DIGIT #{idx}{RESET}")
    print(f"{'='*60}")

    # Step 1: Show the digit
    print()
    print(f"  {BOLD}STEP 1: The handwritten digit{RESET}")
    print()
    print(render_digit(pixels, true_label))

    # Step 2: Show input voltages
    print()
    print(f"  {BOLD}STEP 2: Convert to input voltages{RESET}")
    print()
    print(render_voltages_input(pixels))

    # Step 3: Show circuit architecture
    if show_circuit:
        print()
        print(f"  {BOLD}STEP 3: Feed into the analog circuit{RESET}")
        print()
        print(render_circuit_info())

    # Step 4: Run ngspice
    print()
    print(f"  {BOLD}STEP {'4' if show_circuit else '3'}: Run ngspice (analog SPICE simulation){RESET}")
    print()
    print(f"  {DIM}Generating circuit: 4,736 resistors + 32 diodes...{RESET}")

    outputs, sim_info = run_analog(pixels, true_label)

    if outputs is None:
        print(f"  {RED}Simulation failed: {sim_info}{RESET}")
        return None

    sim_time = sim_info
    print(f"  {DIM}Running ngspice DC operating point analysis...{RESET}")
    print(f"  {GREEN}Done in {sim_time:.2f}s{RESET}")

    # Step 5: Show output voltages
    analog_pred = int(np.argmax(outputs))
    max_val = np.max(outputs)

    print()
    step = '5' if show_circuit else '4'
    print(f"  {BOLD}STEP {step}: Read output voltages{RESET}")
    print(f"  {DIM}Each output node represents one digit (0-9).{RESET}")
    print(f"  {DIM}The highest voltage is the circuit's answer.{RESET}")
    print()

    for j in range(OUTPUT_DIM):
        is_winner = (j == analog_pred)
        is_true = (j == true_label)
        print(render_output_bar(j, outputs[j], max_val, is_winner, is_true and not is_winner))

    # Step 6: Compare with digital
    h1, d_logits = digital_forward(pixels, W1, b1, W2, b2)
    d_pred = int(np.argmax(d_logits))

    # Compute cosine similarity
    norm_a = np.linalg.norm(outputs)
    norm_d = np.linalg.norm(d_logits)
    cos_sim = float(np.dot(outputs, d_logits) / (norm_a * norm_d)) if norm_a > 0 and norm_d > 0 else 0

    print()
    step = '6' if show_circuit else '5'
    print(f"  {BOLD}STEP {step}: The verdict{RESET}")
    print()

    analog_correct = analog_pred == true_label
    digital_correct = d_pred == true_label
    match = analog_pred == d_pred

    if analog_correct:
        print(f"  {GREEN}{BOLD}  ✓ ANALOG CIRCUIT: digit {analog_pred} (CORRECT){RESET}")
    else:
        print(f"  {RED}{BOLD}  ✗ ANALOG CIRCUIT: digit {analog_pred} (WRONG — true: {true_label}){RESET}")

    if digital_correct:
        print(f"  {GREEN}  ✓ Digital numpy:  digit {d_pred} (correct){RESET}")
    else:
        print(f"  {RED}  ✗ Digital numpy:  digit {d_pred} (wrong — true: {true_label}){RESET}")

    print()
    if match:
        print(f"  {CYAN}Analog = Digital? YES (both say {analog_pred}){RESET}")
    else:
        print(f"  {YELLOW}Analog = Digital? NO (analog={analog_pred}, digital={d_pred}){RESET}")
    print(f"  {DIM}Cosine similarity: {cos_sim:.4f} (1.0 = perfect fidelity){RESET}")
    print(f"  {DIM}Simulation time:   {sim_time:.2f}s{RESET}")

    return {
        "true": true_label,
        "analog_pred": analog_pred,
        "digital_pred": d_pred,
        "analog_correct": analog_correct,
        "match": match,
        "cos_sim": cos_sim,
    }


def main():
    parser = argparse.ArgumentParser(description="Analog MNIST demo — watch a circuit classify digits.")
    parser.add_argument("--n", type=int, default=5, help="Number of digits to demo (default: 5)")
    parser.add_argument("--digit-index", type=int, default=None, help="Specific test digit index")
    parser.add_argument("--all", action="store_true", help="Test all digits")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for digit selection")
    args = parser.parse_args()

    X_test, y_test, W1, b1, W2, b2 = load_data()

    print()
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}   ANALOG MNIST: A Circuit That Sees{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    print()
    print(f"  {DIM}A resistive crossbar circuit classifies handwritten digits.{RESET}")
    print(f"  {DIM}No CPU. No GPU. No code. Just physics.{RESET}")
    print()
    print(f"  {DIM}How it works:{RESET}")
    print(f"  {DIM}  1. Each pixel becomes a voltage (0-1V){RESET}")
    print(f"  {DIM}  2. Voltages flow through 4,736 resistors{RESET}")
    print(f"  {DIM}  3. Ohm's law multiplies (V×G = I){RESET}")
    print(f"  {DIM}  4. Kirchhoff's law sums (ΣI = 0){RESET}")
    print(f"  {DIM}  5. Diodes activate (ReLU){RESET}")
    print(f"  {DIM}  6. Highest output voltage = answer{RESET}")
    print()
    print(f"  {DIM}Architecture: 64 → 32 (ReLU) → 10{RESET}")
    print(f"  {DIM}Components:  4,736 resistors + 32 diodes{RESET}")
    print(f"  {DIM}Weights:     2,410 (trained digitally, mapped to conductances){RESET}")

    # Select digits
    if args.digit_index is not None:
        indices = [args.digit_index]
    elif args.all:
        indices = list(range(len(X_test)))
    else:
        rng = np.random.RandomState(args.seed)
        indices = rng.choice(len(X_test), size=args.n, replace=False).tolist()

    results = []
    for i, idx in enumerate(indices):
        show_circuit = (i == 0)  # Only show circuit diagram for first digit
        r = demo_single(idx, X_test[idx], int(y_test[idx]), W1, b1, W2, b2, show_circuit)
        if r:
            results.append(r)

    # Final summary
    if len(results) > 1:
        analog_correct = sum(1 for r in results if r["analog_correct"])
        digital_correct = sum(1 for r in results if r["true"] == r["digital_pred"])
        matches = sum(1 for r in results if r["match"])
        avg_cos = np.mean([r["cos_sim"] for r in results])

        print()
        print(f"{'='*60}")
        print(f"  {BOLD}FINAL SCOREBOARD{RESET}")
        print(f"{'='*60}")
        print()
        print(f"  Digits tested:       {len(results)}")
        print(f"  {BOLD}Analog accuracy:     {GREEN}{analog_correct}/{len(results)} ({100*analog_correct/len(results):.1f}%){RESET}")
        print(f"  Digital accuracy:    {digital_correct}/{len(results)} ({100*digital_correct/len(results):.1f}%)")
        print(f"  Analog=Digital:      {matches}/{len(results)} ({100*matches/len(results):.1f}%)")
        print(f"  Avg cosine sim:      {avg_cos:.4f}")
        print()

        if analog_correct == len(results):
            print(f"  {GREEN}{BOLD}  PERFECT SCORE — every digit classified by pure physics!{RESET}")
        elif analog_correct / len(results) >= 0.9:
            print(f"  {GREEN}{BOLD}  Excellent — the circuit sees with >90% accuracy!{RESET}")
        elif analog_correct / len(results) >= 0.7:
            print(f"  {YELLOW}{BOLD}  Good — room for optimization.{RESET}")
        else:
            print(f"  {RED}{BOLD}  Needs work — autoresearcher time!{RESET}")

        print()
        print(f"  {DIM}Each prediction was computed by 4,736 physical resistors{RESET}")
        print(f"  {DIM}and 32 diodes. No CPU executed any instruction.{RESET}")
        print(f"  {DIM}This is how real analog AI accelerators work.{RESET}")
        print(f"{'='*60}")
        print()


if __name__ == "__main__":
    main()
