#!/usr/bin/env python3
"""
Convert trained weights to ngspice resistive crossbar circuits.

Generates a circuit where:
  - 64 input voltages (0-1V) represent 8x8 pixel intensities
  - Layer 1: 64x32 differential resistive crossbar (4,096 resistors)
  - ReLU: diode-based activation circuits (32 diodes)
  - Layer 2: 32x10 differential resistive crossbar (640 resistors)
  - Winner-take-all: highest output voltage = predicted digit

The circuit performs neural network inference using ONLY analog physics:
  - Ohm's law: V/R = I  (multiplication)
  - KCL: sum of currents at a node = 0  (summation)
  - Diode: passes positive, blocks negative (ReLU)

Usage:
    python generate_circuit.py --digit-index 0       # First test digit
    python generate_circuit.py --pixel-values 0,0,1,0.5,...  # Raw pixel values
    python generate_circuit.py --all-test             # Generate for all test digits
"""

import argparse
import json
import os
import re
import sys
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Analog design parameters (these are what the autoresearcher tunes)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "G_scale": 1e-3,       # Conductance scale (weight=1 maps to this conductance)
    "G_min": 1e-9,         # Minimum conductance (prevents infinite resistance)
    "R_min": 1e3,          # Minimum resistance (1k, convergence floor)
    "R_max": 1e9,          # Maximum resistance (1G, near-zero weight)
    "R_tia": 1e3,          # Transimpedance feedback resistance
    "R_pulldown": 1e6,     # ReLU pull-down resistance
    "V_high": 1.0,         # Input voltage for pixel intensity = 1.0
    "diode_is": 1e-14,     # Diode saturation current (realistic silicon)
    "diode_n": 1.0,        # Diode ideality factor (1.0 = real silicon, 0.001 = ideal)
    "reltol": 1e-4,        # ngspice convergence tolerance
    "abstol": 1e-12,       # ngspice absolute tolerance
    "gmin": 1e-12,         # ngspice minimum conductance
    "mismatch_pct": 5.0,   # Resistor mismatch (% standard deviation, 0 = perfect)
    "mismatch_seed": 42,   # Random seed for reproducible mismatch
}


def load_params(params_path=None):
    """Load analog design parameters, using defaults for any missing."""
    params = DEFAULT_PARAMS.copy()
    if params_path and os.path.exists(params_path):
        with open(params_path, "r") as f:
            overrides = json.load(f)
        params.update(overrides)
    return params


def load_weights(weights_path=None):
    """Load neural network weights from JSON."""
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


def weight_to_conductances(w, params):
    """Map a signed weight to differential conductance pair (G+, G-)."""
    G_scale = params["G_scale"]
    G_min = params["G_min"]

    g_offset = max(abs(w) * G_scale / 2 + G_min, G_min * 2)
    g_plus = g_offset + w * G_scale / 2
    g_minus = g_offset - w * G_scale / 2

    g_plus = max(g_plus, G_min)
    g_minus = max(g_minus, G_min)

    return g_plus, g_minus


def fmt_r(r):
    """Format resistance value for ngspice."""
    if r >= 1e9:
        return f"{r / 1e9:.4g}G"
    elif r >= 1e6:
        return f"{r / 1e6:.4g}Meg"
    elif r >= 1e3:
        return f"{r / 1e3:.4g}k"
    else:
        return f"{r:.4g}"


# ---------------------------------------------------------------------------
# Netlist generation
# ---------------------------------------------------------------------------
def generate_netlist(weights, pixel_values, params, digit_label=None, mode="crossbar"):
    """
    Generate ngspice netlist for analog digit classification.

    Args:
        weights: dict with W1, b1, W2, b2
        pixel_values: array of 64 pixel intensities (0-1)
        params: analog design parameters
        digit_label: optional true label for annotation
        mode: "crossbar" (real resistors) or "hybrid" (B-source matmul)

    Returns:
        netlist: str, complete ngspice netlist
        stats: dict with circuit statistics
    """
    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]
    n_in = W1.shape[0]
    n_hidden = W1.shape[1]
    n_out = W2.shape[1]

    V_high = params["V_high"]
    R_min = params["R_min"]
    R_max = params["R_max"]
    R_tia = params["R_tia"]
    R_pulldown = params["R_pulldown"]
    mismatch_pct = params.get("mismatch_pct", 0.0)
    mismatch_seed = int(params.get("mismatch_seed", 42))

    # Mismatch random generator
    mismatch_rng = np.random.RandomState(mismatch_seed) if mismatch_pct > 0 else None

    def apply_mismatch(r_val):
        """Apply random resistor mismatch (Gaussian, clamp to positive)."""
        if mismatch_rng is None or mismatch_pct <= 0:
            return r_val
        factor = 1.0 + mismatch_rng.normal(0, mismatch_pct / 100.0)
        return max(R_min, r_val * max(factor, 0.5))  # clamp to at least 50% of nominal

    lines = []
    resistor_count = 0
    diode_count = 0

    def emit(s=""):
        lines.append(s)

    # Header
    label_str = f"  True label: {digit_label}" if digit_label is not None else ""
    mismatch_str = f"  Mismatch: {mismatch_pct}% sigma" if mismatch_pct > 0 else "  Mismatch: none (ideal)"
    emit(f"* Analog MNIST Digit Classifier (Resistive Crossbar)")
    emit(f"* ===================================================")
    emit(f"* AUTO-GENERATED by generate_circuit.py")
    emit(f"* Architecture: {n_in} -> {n_hidden} (ReLU) -> {n_out}")
    emit(f"* Mode: {mode}")
    emit(f"* Diode: n={params['diode_n']}, Is={params['diode_is']:.1e}")
    emit(f"*{mismatch_str}")
    emit(f"*{label_str}")
    emit(f"* Resistive crossbar: Ohm's law = multiply, KCL = sum")
    emit(f"*")
    emit()

    # ----- INPUT STAGE -----
    emit(f"* === INPUT: 64 pixel voltages (8x8 grayscale image) ===")
    active_count = 0
    for i in range(n_in):
        v = float(pixel_values[i]) * V_high
        row, col = i // 8, i % 8
        active = " *<< ACTIVE" if v > 0.01 else ""
        emit(f"Vin_{i}  in_{i}  0  dc {v:.4f}  $ pixel[{row}][{col}]{active}")
        if v > 0.01:
            active_count += 1
    emit(f"* Active pixels: {active_count}/64")
    emit()

    if mode == "crossbar":
        # ----- LAYER 1 CROSSBAR -----
        emit(f"* === LAYER 1: {n_in}x{n_hidden} Resistive Crossbar ===")
        emit(f"* Each weight -> differential resistor pair (G+, G-)")
        emit(f"* Virtual ground (0V source) at each column")
        emit()

        for j in range(n_hidden):
            emit(f"* --- Hidden neuron {j} ---")
            emit(f"V_vg1p_{j}  sum1p_{j}  0  dc 0")
            emit(f"V_vg1n_{j}  sum1n_{j}  0  dc 0")

            for i in range(n_in):
                w = W1[i][j]
                g_plus, g_minus = weight_to_conductances(w, params)
                r_plus = apply_mismatch(max(R_min, min(1.0 / g_plus, R_max)))
                r_minus = apply_mismatch(max(R_min, min(1.0 / g_minus, R_max)))

                emit(f"R1p_{i}_{j}  in_{i}  sum1p_{j}  {fmt_r(r_plus)}")
                emit(f"R1n_{i}_{j}  in_{i}  sum1n_{j}  {fmt_r(r_minus)}")
                resistor_count += 2

            bias_v = b1[j]
            emit(f"B_h1r_{j}  h1_raw_{j}  0  "
                 f"V = (i(V_vg1p_{j}) - i(V_vg1n_{j})) * {R_tia:.1f}"
                 f" + {bias_v:.6e}")
            emit()

    else:
        # ----- LAYER 1 HYBRID (B-SOURCE) -----
        emit(f"* === LAYER 1: {n_in}->{n_hidden} Behavioral Matmul ===")
        emit()

        for j in range(n_hidden):
            terms = []
            for i in range(n_in):
                w = W1[i][j]
                if abs(w) > 1e-10:
                    terms.append(f"{w:+.6e}*v(in_{i})")

            bias = b1[j]
            if terms:
                expr = " ".join(terms)
                if abs(bias) > 1e-10:
                    expr += f" {bias:+.6e}"
            else:
                expr = f"{bias:.6e}"

            full_expr = f"V = {expr}"
            if len(full_expr) < 900:
                emit(f"B_h1r_{j}  h1_raw_{j}  0  {full_expr}")
            else:
                emit(f"B_h1r_{j}  h1_raw_{j}  0")
                all_terms = re.findall(r'[+-][^+-]+', expr)
                line = ""
                first = True
                for term in all_terms:
                    if len(line) + len(term) > 800:
                        if first:
                            emit(f"+ V = {line}")
                            first = False
                        else:
                            emit(f"+ {line}")
                        line = term
                    else:
                        line += term
                if line:
                    if first:
                        emit(f"+ V = {line}")
                    else:
                        emit(f"+ {line}")
            emit()

    # ----- RELU -----
    emit(f"* === ReLU ACTIVATION (diode circuits) ===")
    emit(f".model diode_relu d(is={params['diode_is']:.2e} n={params['diode_n']} bv=100 ibv=1e-10)")
    emit()
    for j in range(n_hidden):
        emit(f"D_relu_{j}  h1_raw_{j}  h1_{j}  diode_relu")
        emit(f"R_pd_{j}  h1_{j}  0  {fmt_r(R_pulldown)}")
        diode_count += 1
    emit()

    if mode == "crossbar":
        # ----- LAYER 2 CROSSBAR -----
        emit(f"* === LAYER 2: {n_hidden}x{n_out} Resistive Crossbar ===")
        emit()

        for j in range(n_out):
            emit(f"* --- Output neuron {j} (digit '{j}') ---")
            emit(f"V_vg2p_{j}  sum2p_{j}  0  dc 0")
            emit(f"V_vg2n_{j}  sum2n_{j}  0  dc 0")

            for i in range(n_hidden):
                w = W2[i][j]
                g_plus, g_minus = weight_to_conductances(w, params)
                r_plus = apply_mismatch(max(R_min, min(1.0 / g_plus, R_max)))
                r_minus = apply_mismatch(max(R_min, min(1.0 / g_minus, R_max)))

                emit(f"R2p_{i}_{j}  h1_{i}  sum2p_{j}  {fmt_r(r_plus)}")
                emit(f"R2n_{i}_{j}  h1_{i}  sum2n_{j}  {fmt_r(r_minus)}")
                resistor_count += 2

            bias_v = b2[j]
            emit(f"B_out_{j}  out_{j}  0  "
                 f"V = (i(V_vg2p_{j}) - i(V_vg2n_{j})) * {R_tia:.1f}"
                 f" + {bias_v:.6e}")
            emit()

    else:
        # ----- LAYER 2 HYBRID -----
        emit(f"* === LAYER 2: {n_hidden}->{n_out} Behavioral Matmul ===")
        emit()

        for j in range(n_out):
            terms = []
            for i in range(n_hidden):
                w = W2[i][j]
                if abs(w) > 1e-10:
                    terms.append(f"{w:+.6e}*v(h1_{i})")

            bias = b2[j]
            if terms:
                expr = " ".join(terms)
                if abs(bias) > 1e-10:
                    expr += f" {bias:+.6e}"
            else:
                expr = f"{bias:.6e}"

            full_expr = f"V = {expr}"
            if len(full_expr) < 900:
                emit(f"B_out_{j}  out_{j}  0  {full_expr}")
            else:
                emit(f"B_out_{j}  out_{j}  0")
                all_terms = re.findall(r'[+-][^+-]+', expr)
                line = ""
                first = True
                for term in all_terms:
                    if len(line) + len(term) > 800:
                        if first:
                            emit(f"+ V = {line}")
                            first = False
                        else:
                            emit(f"+ {line}")
                        line = term
                    else:
                        line += term
                if line:
                    if first:
                        emit(f"+ V = {line}")
                    else:
                        emit(f"+ {line}")
            emit()

    # ----- WINNER-TAKE-ALL -----
    emit(f"* === WINNER-TAKE-ALL (max output = predicted digit) ===")
    current_terms = [f"v(out_{j})" for j in range(n_out)]
    while len(current_terms) > 1:
        next_terms = []
        for k in range(0, len(current_terms), 2):
            if k + 1 < len(current_terms):
                next_terms.append(f"max({current_terms[k]},{current_terms[k+1]})")
            else:
                next_terms.append(current_terms[k])
        current_terms = next_terms
    emit(f"B_max  vmax  0  V = {current_terms[0]}")
    emit()
    for j in range(n_out):
        emit(f"B_win_{j}  win_{j}  0  "
             f"V = (v(out_{j}) >= v(vmax) - 0.01) ? 1.0 : 0.0  $ digit {j}")
    emit()

    # ----- SIMULATION CONTROL -----
    emit(f"* === SIMULATION ===")
    emit(f".options reltol={params['reltol']:.1e} abstol={params['abstol']:.1e} "
         f"vntol=1e-6 gmin={params['gmin']:.1e}")
    emit(f".options itl1=500 itl2=200")
    emit()
    emit(f".control")
    emit(f"set filetype=ascii")
    emit(f"op")
    emit()

    # Print outputs
    emit(f"echo \"=== OUTPUT VOLTAGES (digit scores) ===\"")
    for j in range(n_out):
        emit(f"echo \"out_{j} (digit {j}) = $&v(out_{j})\"")
    emit()
    emit(f"echo \"\"")
    emit(f"echo \"=== PREDICTION ===\"")
    emit(f"echo \"max_output = $&v(vmax)\"")
    for j in range(n_out):
        emit(f"echo \"win_{j} (digit {j}) = $&v(win_{j})\"")
    emit()

    # Save data
    out_vars = " ".join(f"v(out_{j})" for j in range(n_out))
    h1_vars = " ".join(f"v(h1_{j})" for j in range(n_hidden))
    emit(f"wrdata analog_output.txt {out_vars}")
    emit(f"wrdata analog_hidden.txt {h1_vars}")
    emit()
    if digit_label is not None:
        emit(f"echo \"\"")
        emit(f"echo \"TRUE LABEL: {digit_label}\"")
    emit(f"echo \"Simulation complete.\"")
    emit(f"quit")
    emit(f".endc")
    emit()
    emit(f".end")

    netlist = "\n".join(lines)
    stats = {
        "resistors": resistor_count,
        "diodes": diode_count,
        "nodes": n_in + 2 * n_hidden + 2 * n_out + 1 + (4 * n_hidden + 4 * n_out if mode == "crossbar" else 0),
        "active_pixels": active_count,
        "mode": mode,
        "lines": len(lines),
    }
    return netlist, stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate analog MNIST classifier circuit.")
    parser.add_argument("--digit-index", "-d", type=int, default=0,
                        help="Index into test set (default: 0)")
    parser.add_argument("--pixel-values", type=str, default=None,
                        help="Comma-separated pixel values (0-1), length must match weights input dim")
    parser.add_argument("--label", type=int, default=None,
                        help="True label (for annotation)")
    parser.add_argument("--mode", choices=["crossbar", "hybrid"], default="crossbar",
                        help="Circuit mode (default: crossbar)")
    parser.add_argument("--params", type=str, default=None,
                        help="Path to analog parameters JSON")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights JSON")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output netlist path")
    parser.add_argument("--all-test", action="store_true",
                        help="Generate circuits for first 100 test digits")
    args = parser.parse_args()

    # Load weights and params
    weights = load_weights(args.weights)
    params = load_params(args.params)

    n_in = weights["W1"].shape[0]
    img_size = weights.get("architecture", {}).get("img_size", int(np.sqrt(n_in)))
    mode_ds = "8x8" if img_size == 8 else "14x14"

    if args.pixel_values:
        pixels = np.array([float(x) for x in args.pixel_values.split(",")])
        if len(pixels) != n_in:
            print(f"ERROR: Expected {n_in} pixel values, got {len(pixels)}")
            sys.exit(1)
        label = args.label

        netlist, stats = generate_netlist(weights, pixels, params, label, args.mode)
        out_path = args.output or str(SCRIPT_DIR / "analog_mnist.cir")
        with open(out_path, "w", newline="\n") as f:
            f.write(netlist)
        print(f"Written: {out_path} ({stats['lines']} lines, {stats['resistors']} resistors)")

    elif args.all_test:
        # Load test set
        sys.path.insert(0, str(SCRIPT_DIR))
        from train_model import load_dataset, train_test_split
        X, y, _ = load_dataset(mode_ds)
        _, _, X_test, y_test = train_test_split(X, y)

        out_dir = SCRIPT_DIR / "circuits"
        out_dir.mkdir(exist_ok=True)

        n = min(100, len(X_test))
        for idx in range(n):
            netlist, stats = generate_netlist(
                weights, X_test[idx], params, int(y_test[idx]), args.mode
            )
            path = out_dir / f"digit_{idx:03d}_label{y_test[idx]}.cir"
            with open(path, "w", newline="\n") as f:
                f.write(netlist)
            if idx % 10 == 0:
                print(f"  Generated {idx+1}/{n}...")
        print(f"Done: {n} circuits in {out_dir}/")

    else:
        # Single digit from test set
        sys.path.insert(0, str(SCRIPT_DIR))
        from train_model import load_dataset, train_test_split
        X, y, _ = load_dataset(mode_ds)
        _, _, X_test, y_test = train_test_split(X, y)

        idx = args.digit_index
        if idx >= len(X_test):
            print(f"ERROR: Index {idx} out of range (max {len(X_test)-1})")
            sys.exit(1)

        pixels = X_test[idx]
        label = int(y_test[idx])

        netlist, stats = generate_netlist(weights, pixels, params, label, args.mode)
        out_path = args.output or str(SCRIPT_DIR / "analog_mnist.cir")
        with open(out_path, "w", newline="\n") as f:
            f.write(netlist)

        # Show the digit
        print(f"\nDigit #{idx} (true label: {label})")
        img = pixels.reshape(img_size, img_size)
        chars = " .:-=+*#@"
        print("  +--------+")
        for row in img:
            line = ""
            for val in row:
                ci = min(int(val * (len(chars) - 1)), len(chars) - 1)
                line += chars[ci]
            print(f"  |{line}|")
        print("  +--------+")
        print(f"\nWritten: {out_path}")
        print(f"  Mode: {stats['mode']}")
        print(f"  Resistors: {stats['resistors']}")
        print(f"  Diodes: {stats['diodes']}")
        print(f"  Nodes: ~{stats['nodes']}")
        print(f"  Lines: {stats['lines']}")
        print(f"\nRun: ngspice -b {out_path}")


if __name__ == "__main__":
    main()
