#!/usr/bin/env python3
"""
Neural Network to Analog ngspice Netlist Generator
===================================================

Reads weights_analog.json (exported by train_model.py) and generates ngspice
netlists that perform neural network inference using analog circuits.

Architecture:
  - 3-character input (context window) -> one-hot encoding (81 inputs)
  - Layer 1: 81 -> 32 (hidden layer with ReLU activation)
  - Layer 2: 32 -> 27 (output layer, 27 = size of alphabet including space)

Generates TWO netlists:
  1. analog_nn.cir         -- Hybrid: behavioral matmul + real analog activation
  2. analog_nn_crossbar.cir -- Full resistive crossbar with differential columns

Input encoding:
  Characters: ' abcdefghijklmnopqrstuvwxyz' (27 symbols, space=0, a=1, ..., z=26)
  3 characters -> 3 x 27 = 81 one-hot inputs
  in_0..in_26   = first character
  in_27..in_53  = second character
  in_54..in_80  = third character

Usage:
  python generate_netlist.py                           # Default input "hel"
  python generate_netlist.py --input "hel"             # Specific 3-char input
  python generate_netlist.py --input "hel" --full-crossbar  # Also generate crossbar
  python generate_netlist.py --batch "hel,wor,the,neu" # Multiple test cases
  python generate_netlist.py --weights weights.json    # Use alternate weights file
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ALPHABET = " abcdefghijklmnopqrstuvwxyz"  # 27 chars: space + a-z
VOCAB_SIZE = len(ALPHABET)  # 27
CONTEXT_LEN = 3
INPUT_SIZE = CONTEXT_LEN * VOCAB_SIZE  # 81

# Analog circuit parameters
V_HIGH = 1.0           # Voltage representing logic '1' in one-hot encoding
V_LOW = 0.0            # Voltage representing logic '0'
R_MIN = 1e3            # Minimum resistor value (1k) to avoid convergence issues
R_MAX = 1e9            # Maximum resistor value (1G) for near-zero weights
G_MIN = 1.0 / R_MAX    # Minimum conductance
G_SCALE = 1e-3         # Conductance scaling factor (maps weight=1 to G=1mS)
R_TIA = 1e3            # Transimpedance amplifier feedback resistance (1k)
R_PULLDOWN = 1e6       # ReLU pull-down resistor (1M)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def char_to_index(ch: str) -> int:
    """Convert character to alphabet index (space=0, a=1, ..., z=26)."""
    ch = ch.lower()
    idx = ALPHABET.find(ch)
    if idx < 0:
        raise ValueError(f"Character '{ch}' not in alphabet: '{ALPHABET}'")
    return idx


def encode_input(text: str) -> list:
    """Encode a 3-character string as an 81-element one-hot vector."""
    if len(text) != CONTEXT_LEN:
        raise ValueError(
            f"Input must be exactly {CONTEXT_LEN} characters, got {len(text)}: '{text}'"
        )
    vec = [0.0] * INPUT_SIZE
    for pos, ch in enumerate(text):
        idx = char_to_index(ch)
        vec[pos * VOCAB_SIZE + idx] = 1.0
    return vec


def fmt_resistance(r: float) -> str:
    """Format a resistance value in ngspice-friendly notation."""
    if r >= 1e9:
        return f"{r / 1e9:.4g}G"
    elif r >= 1e6:
        return f"{r / 1e6:.4g}Meg"
    elif r >= 1e3:
        return f"{r / 1e3:.4g}k"
    else:
        return f"{r:.4g}"


def weight_to_conductances(w: float) -> tuple:
    """
    Map a signed weight to differential conductance pair (G_plus, G_minus).

    Uses the standard differential mapping:
      G_plus  = G_offset + w * G_SCALE / 2
      G_minus = G_offset - w * G_SCALE / 2

    where G_offset is chosen so both conductances are always positive.
    The effective weight is proportional to (G_plus - G_minus).
    """
    g_offset = max(abs(w) * G_SCALE / 2 + G_MIN, G_MIN * 2)
    g_plus = g_offset + w * G_SCALE / 2
    g_minus = g_offset - w * G_SCALE / 2

    # Clamp to physical bounds
    g_plus = max(g_plus, G_MIN)
    g_minus = max(g_minus, G_MIN)

    return g_plus, g_minus


def load_weights(weights_path: str) -> dict:
    """
    Load neural network weights from JSON file.

    Expected format (from train_model.py):
    {
        "metadata": { ... },
        "layer1": {
            "weights": [[...], ...],   // shape [81][32]
            "biases": [...]             // shape [32]
        },
        "layer2": {
            "weights": [[...], ...],   // shape [32][27]
            "biases": [...]             // shape [27]
        }
    }

    Also supports flat format:
    {
        "W1": [[...], ...],  "b1": [...],
        "W2": [[...], ...],  "b2": [...]
    }
    """
    with open(weights_path, "r") as f:
        data = json.load(f)

    # Normalize to canonical format
    if "layer1" in data:
        W1 = data["layer1"]["weights"]
        b1 = data["layer1"]["biases"]
        W2 = data["layer2"]["weights"]
        b2 = data["layer2"]["biases"]
    elif "W1" in data:
        W1 = data["W1"]
        b1 = data["b1"]
        W2 = data["W2"]
        b2 = data["b2"]
    else:
        raise ValueError(
            "Unrecognized weights format. Expected 'layer1'/'layer2' or 'W1'/'W2' keys."
        )

    # Validate shapes
    n_in = len(W1)
    n_hidden = len(W1[0])
    n_out = len(W2[0])

    if n_in != INPUT_SIZE:
        raise ValueError(f"W1 has {n_in} inputs, expected {INPUT_SIZE}")
    if len(W2) != n_hidden:
        raise ValueError(f"W2 has {len(W2)} inputs, expected {n_hidden} (hidden size)")
    if len(b1) != n_hidden:
        raise ValueError(f"b1 has {len(b1)} elements, expected {n_hidden}")
    if len(b2) != n_out:
        raise ValueError(f"b2 has {len(b2)} elements, expected {n_out}")

    metadata = data.get("metadata", {})

    return {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "n_in": n_in,
        "n_hidden": n_hidden,
        "n_out": n_out,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# Netlist Generation: Hybrid (behavioral matmul + analog activation)
# ---------------------------------------------------------------------------


def generate_hybrid_netlist(weights: dict, input_vec: list, input_text: str) -> str:
    """
    Generate the hybrid analog_nn.cir netlist.

    - Matrix multiplications use behavioral voltage sources (B-sources)
    - ReLU activation uses near-ideal diode circuits
    - Winner-take-all output uses behavioral max detection
    - DC operating point analysis for static inference
    """
    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]
    n_in = weights["n_in"]
    n_hidden = weights["n_hidden"]
    n_out = weights["n_out"]

    lines = []

    def emit(s=""):
        lines.append(s)

    # -----------------------------------------------------------------------
    # Title and header
    # -----------------------------------------------------------------------
    emit(f"* Analog Neural Network Inference Circuit (Hybrid)")
    emit(f"* =================================================")
    emit(f"* AUTO-GENERATED by generate_netlist.py")
    emit(f"* Input text: '{input_text}'")
    emit(f"* Architecture: {n_in} -> {n_hidden} (ReLU) -> {n_out} (softmax approx)")
    emit(f"* Matmul: behavioral voltage sources (B-sources)")
    emit(f"* Activation: real diode-based ReLU circuits")
    emit(f"* Output: analog winner-take-all comparator")
    emit(f"*")
    emit(f"* Alphabet: '{ALPHABET}' (27 symbols)")
    emit(f"* Input encoding: one-hot, {V_HIGH}V = active, {V_LOW}V = inactive")
    emit(f"*")

    # -----------------------------------------------------------------------
    # Section 1: Input voltage sources (one-hot encoded)
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* INPUT STAGE: One-hot encoded character voltages")
    emit(f"* ===========================================================")
    emit(f"* {CONTEXT_LEN} characters x {VOCAB_SIZE} one-hot values = {n_in} inputs")
    emit(f"* Input text: '{input_text}'")

    for pos, ch in enumerate(input_text):
        idx = char_to_index(ch)
        emit(f"* Character {pos} = '{ch}' (index {idx}), "
             f"one-hot at position {pos * VOCAB_SIZE + idx}")

    emit()

    active_count = 0
    for i in range(n_in):
        v = input_vec[i] * V_HIGH
        char_pos = i // VOCAB_SIZE
        char_idx = i % VOCAB_SIZE
        char_label = ALPHABET[char_idx] if char_idx < len(ALPHABET) else "?"
        active = " *<< ACTIVE" if v > 0 else ""
        emit(f"Vin_{i}  in_{i}  0  dc {v:.1f}"
             f"  $ char{char_pos}='{char_label}' idx={char_idx}{active}")
        if v > 0:
            active_count += 1

    emit(f"* Total active inputs: {active_count}")
    emit()

    # -----------------------------------------------------------------------
    # Section 2: Layer 1 - behavioral matmul (dot product + bias)
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* LAYER 1: {n_in} -> {n_hidden} Matrix Multiply (Behavioral)")
    emit(f"* ===========================================================")
    emit(f"* Each neuron computes: h1_raw_j = sum(W1[i][j] * V(in_i)) + b1[j]")
    emit(f"* Using ngspice B-source (behavioral voltage source)")
    emit()

    for j in range(n_hidden):
        # Build the weighted sum expression
        # Only include non-zero terms for readability and efficiency
        terms = []
        for i in range(n_in):
            w = W1[i][j]
            if abs(w) > 1e-10:
                terms.append(f"{w:+.6e}*v(in_{i})")

        bias = b1[j]

        if terms:
            # ngspice has a line length limit; split very long expressions
            # across continuation lines using '+'
            expr = " ".join(terms)
            if abs(bias) > 1e-10:
                expr += f" {bias:+.6e}"
            elif not terms:
                expr = f"{bias:.6e}"
        else:
            expr = f"{bias:.6e}"

        # Write as B-source. Use continuation lines for long expressions.
        emit(f"* Hidden neuron {j}: {len(terms)} non-zero weights, bias={bias:.4f}")

        # ngspice B-source line length limit is ~1024 chars.
        # We split the expression into manageable chunks.
        full_expr = f"V = {expr}"

        if len(full_expr) < 900:
            emit(f"B_h1r_{j}  h1_raw_{j}  0  {full_expr}")
        else:
            # Split into multiple continuation lines at term boundaries
            emit(f"B_h1r_{j}  h1_raw_{j}  0")
            # Split terms: each term looks like +X.XXe+XX*v(in_N) or -X.XXe+XX*v(in_N)
            import re
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

    # -----------------------------------------------------------------------
    # Section 3: Layer 1 - ReLU activation (real analog diode circuit)
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* LAYER 1: ReLU Activation (Analog Diode Circuit)")
    emit(f"* ===========================================================")
    emit(f"* ReLU(x) = max(0, x)")
    emit(f"* Implementation: near-ideal diode passes positive voltages,")
    emit(f"* pull-down resistor ensures output = 0V when diode is off.")
    emit(f"*")
    emit(f"* Circuit per neuron:")
    emit(f"*   h1_raw_j ---[D]---> h1_j")
    emit(f"*                        |")
    emit(f"*                      [R_pd]")
    emit(f"*                        |")
    emit(f"*                       GND")
    emit(f"*")
    emit(f"* The diode model uses very low ideality factor (n=0.001)")
    emit(f"* for near-ideal switching behavior (sharp knee at 0V).")
    emit()

    emit(f".model diode_relu d(is=1e-15 n=0.001 bv=100 ibv=1e-10)")
    emit()

    for j in range(n_hidden):
        emit(f"D_relu_{j}  h1_raw_{j}  h1_{j}  diode_relu  $ ReLU for hidden neuron {j}")
        emit(f"R_pd_{j}  h1_{j}  0  {fmt_resistance(R_PULLDOWN)}  $ pull-down")

    emit()

    # -----------------------------------------------------------------------
    # Section 4: Layer 2 - behavioral matmul (dot product + bias)
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* LAYER 2: {n_hidden} -> {n_out} Matrix Multiply (Behavioral)")
    emit(f"* ===========================================================")
    emit(f"* Each output neuron computes: out_raw_j = sum(W2[i][j] * V(h1_i)) + b2[j]")
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

        char_label = ALPHABET[j] if j < len(ALPHABET) else "?"
        emit(f"* Output neuron {j} ('{char_label}'): "
             f"{len(terms)} non-zero weights, bias={bias:.4f}")

        full_expr = f"V = {expr}"

        if len(full_expr) < 900:
            emit(f"B_out_{j}  out_{j}  0  {full_expr}")
        else:
            emit(f"B_out_{j}  out_{j}  0")
            import re
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

    # -----------------------------------------------------------------------
    # Section 5: Winner-take-all output (analog comparator)
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* OUTPUT STAGE: Winner-Take-All (Analog Comparator Tree)")
    emit(f"* ===========================================================")
    emit(f"* Finds the output neuron with the highest voltage.")
    emit(f"* B_max computes the maximum output voltage.")
    emit(f"* B_winner_j outputs 1V if neuron j is the winner, 0V otherwise.")
    emit()

    # Build a nested max expression
    # ngspice behavioral sources support max(a, b)
    # We build a binary tree: max(max(max(a,b), c), d) ...
    if n_out == 1:
        max_expr = "v(out_0)"
    else:
        # Build pairwise max tree
        current_terms = [f"v(out_{j})" for j in range(n_out)]
        while len(current_terms) > 1:
            next_terms = []
            for k in range(0, len(current_terms), 2):
                if k + 1 < len(current_terms):
                    next_terms.append(f"max({current_terms[k]},{current_terms[k+1]})")
                else:
                    next_terms.append(current_terms[k])
            current_terms = next_terms
        max_expr = current_terms[0]

    emit(f"* Maximum output voltage (nested binary max tree)")
    emit(f"B_max  vmax  0  V = {max_expr}")
    emit()

    emit(f"* Winner detection: 1V if this neuron has the max output, 0V otherwise")
    emit(f"* Uses a small tolerance window (10mV) for comparison")
    for j in range(n_out):
        char_label = ALPHABET[j] if j < len(ALPHABET) else "?"
        emit(f"B_win_{j}  win_{j}  0  "
             f"V = (v(out_{j}) >= v(vmax) - 0.01) ? {V_HIGH} : {V_LOW}"
             f"  $ '{char_label}'")

    emit()

    # -----------------------------------------------------------------------
    # Section 6: Simulation control
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* SIMULATION CONTROL")
    emit(f"* ===========================================================")
    emit(f"* DC operating point analysis for static inference.")
    emit(f"* No transient needed -- purely combinational analog circuit.")
    emit()

    # Use .options for convergence
    emit(f".options reltol=1e-4 abstol=1e-12 vntol=1e-6 gmin=1e-12")
    emit()

    emit(f".control")
    emit(f"set filetype=ascii")
    emit()
    emit(f"* DC operating point (static inference)")
    emit(f"op")
    emit()

    # Print hidden layer activations
    emit(f"* === Hidden Layer Activations ===")
    h1_nodes = " ".join(f"v(h1_{j})" for j in range(n_hidden))
    emit(f"echo \"=== HIDDEN LAYER (after ReLU) ===\"")
    for j in range(n_hidden):
        emit(f"echo \"h1_{j} = $&v(h1_{j})\"")

    emit()

    # Print output layer values
    emit(f"* === Output Layer Values ===")
    emit(f"echo \"\"")
    emit(f"echo \"=== OUTPUT LAYER ===\"")
    for j in range(n_out):
        char_label = ALPHABET[j] if j < len(ALPHABET) else "?"
        emit(f"echo \"out_{j} ('{char_label}') = $&v(out_{j})\"")

    emit()

    # Print winner
    emit(f"* === Winner Detection ===")
    emit(f"echo \"\"")
    emit(f"echo \"=== PREDICTED CHARACTER ===\"")
    emit(f"echo \"max_output = $&v(vmax)\"")
    for j in range(n_out):
        char_label = ALPHABET[j] if j < len(ALPHABET) else "?"
        emit(f"echo \"win_{j} ('{char_label}') = $&v(win_{j})\"")

    emit()

    # Save data files for dashboard
    emit(f"* === Save data for dashboard ===")

    # Hidden layer
    h1_vars = " ".join(f"v(h1_{j})" for j in range(n_hidden))
    emit(f"wrdata nn_hidden.txt {h1_vars}")

    # Output layer
    out_vars = " ".join(f"v(out_{j})" for j in range(n_out))
    emit(f"wrdata nn_output.txt {out_vars}")

    # Winner
    win_vars = " ".join(f"v(win_{j})" for j in range(n_out))
    emit(f"wrdata nn_winner.txt {win_vars}")

    emit()
    emit(f"echo \"\"")
    emit(f"echo \"Inference complete for input '{input_text}'.\"")
    emit(f"quit")
    emit(f".endc")
    emit()
    emit(f".end")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Netlist Generation: Full Resistive Crossbar
# ---------------------------------------------------------------------------


def generate_crossbar_netlist(weights: dict, input_vec: list, input_text: str) -> str:
    """
    Generate the full resistive crossbar analog_nn_crossbar.cir netlist.

    - Matrix multiplications use REAL resistors in differential crossbar arrays
    - Each weight is mapped to a differential conductance pair (G+, G-)
    - Inverting summing amplifiers (ideal opamps via VCVS) provide virtual ground
    - ReLU activation uses near-ideal diode circuits
    - Winner-take-all output uses behavioral max detection
    """
    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]
    n_in = weights["n_in"]
    n_hidden = weights["n_hidden"]
    n_out = weights["n_out"]

    lines = []
    resistor_count = 0

    def emit(s=""):
        lines.append(s)

    # -----------------------------------------------------------------------
    # Title and header
    # -----------------------------------------------------------------------
    emit(f"* Analog Neural Network Inference Circuit (Full Resistive Crossbar)")
    emit(f"* ==================================================================")
    emit(f"* AUTO-GENERATED by generate_netlist.py")
    emit(f"* Input text: '{input_text}'")
    emit(f"* Architecture: {n_in} -> {n_hidden} (ReLU) -> {n_out}")
    emit(f"* Matmul: differential resistive crossbar arrays")
    emit(f"* Activation: real diode-based ReLU circuits")
    emit(f"* Output: analog winner-take-all comparator")
    emit(f"*")
    emit(f"* Each weight W[i][j] is mapped to a differential conductance pair:")
    emit(f"*   G+ = G_offset + W * G_SCALE / 2")
    emit(f"*   G- = G_offset - W * G_SCALE / 2")
    emit(f"*   Effective weight proportional to (G+ - G-) = W * G_SCALE")
    emit(f"*")
    emit(f"* Topology per column: inverting summing amplifier with virtual ground")
    emit(f"*   - VCVS (E-source) with gain -1e6 provides virtual ground at summing node")
    emit(f"*   - Feedback resistor R_fb converts summed current to voltage")
    emit(f"*   - Differential pair (pos/neg columns) handles signed weights")
    emit(f"*")

    # -----------------------------------------------------------------------
    # Section 1: Input voltage sources
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* INPUT STAGE: One-hot encoded character voltages")
    emit(f"* ===========================================================")
    emit()

    for i in range(n_in):
        v = input_vec[i] * V_HIGH
        char_pos = i // VOCAB_SIZE
        char_idx = i % VOCAB_SIZE
        char_label = ALPHABET[char_idx] if char_idx < len(ALPHABET) else "?"
        active = " *<< ACTIVE" if v > 0 else ""
        emit(f"Vin_{i}  in_{i}  0  dc {v:.1f}"
             f"  $ char{char_pos}='{char_label}' idx={char_idx}{active}")

    emit()

    # -----------------------------------------------------------------------
    # Section 2: Layer 1 - Resistive crossbar (81 x 32)
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* LAYER 1: {n_in} x {n_hidden} Resistive Crossbar Array")
    emit(f"* ===========================================================")
    emit(f"* Differential crossbar with inverting summing amplifiers.")
    emit(f"*")
    emit(f"* For each output neuron j:")
    emit(f"*   1. Positive column: resistors from each input to sum1p_j")
    emit(f"*   2. Negative column: resistors from each input to sum1n_j")
    emit(f"*   3. Virtual ground at summing nodes (ideal opamp via VCVS)")
    emit(f"*   4. Feedback resistor converts current sum to voltage")
    emit(f"*   5. Differential amplifier: V_out = V_pos - V_neg + bias")
    emit()

    for j in range(n_hidden):
        emit(f"* --- Hidden neuron {j} ---")

        # Virtual ground amplifiers for positive and negative columns
        # E-source: E_name node+ node- ctrl+ ctrl- gain
        # We want the summing node to be held at 0V (virtual ground).
        # Use a high-gain VCVS that drives the summing node:
        #   V(sum_node) = -1e6 * V(sum_node) ... this doesn't work directly.
        #
        # Instead, use a voltage-controlled voltage source to clamp:
        # E_vgp1_j  sum1p_j  0  0  sum1p_j  1e6
        # This creates: V(sum1p_j) = 1e6 * (0 - V(sum1p_j))
        #             => V(sum1p_j) * (1 + 1e6) = 0
        #             => V(sum1p_j) ~ 0  (virtual ground)
        #
        # Actually in ngspice, the proper way is to use an ideal opamp model
        # or a behavioral source for the virtual ground. The cleanest approach
        # is to use a 0V voltage source (which acts as an ammeter) at each
        # column, then measure the current through it.

        # Approach: Use 0V voltage source as virtual ground + ammeter
        # V_vgp1_j  sum1p_j  0  dc 0  (clamps to 0V, we measure I through it)
        # V_vgn1_j  sum1n_j  0  dc 0

        emit(f"V_vg1p_{j}  sum1p_{j}  0  dc 0  $ virtual ground (pos column)")
        emit(f"V_vg1n_{j}  sum1n_{j}  0  dc 0  $ virtual ground (neg column)")
        emit()

        # Resistors from each input to the summing nodes
        for i in range(n_in):
            w = W1[i][j]
            g_plus, g_minus = weight_to_conductances(w)
            r_plus = 1.0 / g_plus
            r_minus = 1.0 / g_minus

            # Clamp resistor values to physical range
            r_plus = max(R_MIN, min(r_plus, R_MAX))
            r_minus = max(R_MIN, min(r_minus, R_MAX))

            emit(f"R_1p_{i}_{j}  in_{i}  sum1p_{j}  {fmt_resistance(r_plus)}"
                 f"  $ W1[{i}][{j}]={w:+.4f} G+={g_plus:.4e}")
            emit(f"R_1n_{i}_{j}  in_{i}  sum1n_{j}  {fmt_resistance(r_minus)}"
                 f"  $ G-={g_minus:.4e}")
            resistor_count += 2

        emit()

        # Differential output: V = (I_pos - I_neg) * R_TIA + bias
        # Current through V_vg1p_j flows FROM sum1p_j TO ground (positive if
        # current flows into the column from the inputs). In ngspice,
        # i(V_vg1p_j) is the current flowing from + to - terminal of the source.
        # Since inputs have positive voltages and the column is at 0V,
        # current flows from input through resistor into the column node,
        # i.e., into the + terminal of V_vg1p_j, so i(V_vg1p_j) > 0.
        #
        # Weighted sum voltage = (I_neg - I_pos) * R_TIA
        # (sign flip because inverting summing amplifier)
        # But we want the same sign convention as the behavioral version,
        # so we use I_pos - I_neg and adjust signs in the weight mapping.

        bias_v = b1[j]
        emit(f"* Transimpedance: I_diff * R_TIA + bias")
        emit(f"B_h1r_{j}  h1_raw_{j}  0  "
             f"V = (i(V_vg1p_{j}) - i(V_vg1n_{j})) * {R_TIA:.1f}"
             f" + {bias_v:.6e}")
        emit()

    # -----------------------------------------------------------------------
    # Section 3: Layer 1 - ReLU activation
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* LAYER 1: ReLU Activation (Analog Diode Circuit)")
    emit(f"* ===========================================================")
    emit()
    emit(f".model diode_relu d(is=1e-15 n=0.001 bv=100 ibv=1e-10)")
    emit()

    for j in range(n_hidden):
        emit(f"D_relu_{j}  h1_raw_{j}  h1_{j}  diode_relu")
        emit(f"R_pd_{j}  h1_{j}  0  {fmt_resistance(R_PULLDOWN)}")

    emit()

    # -----------------------------------------------------------------------
    # Section 4: Layer 2 - Resistive crossbar (32 x 27)
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* LAYER 2: {n_hidden} x {n_out} Resistive Crossbar Array")
    emit(f"* ===========================================================")
    emit()

    for j in range(n_out):
        char_label = ALPHABET[j] if j < len(ALPHABET) else "?"
        emit(f"* --- Output neuron {j} ('{char_label}') ---")

        emit(f"V_vg2p_{j}  sum2p_{j}  0  dc 0  $ virtual ground (pos column)")
        emit(f"V_vg2n_{j}  sum2n_{j}  0  dc 0  $ virtual ground (neg column)")
        emit()

        for i in range(n_hidden):
            w = W2[i][j]
            g_plus, g_minus = weight_to_conductances(w)
            r_plus = 1.0 / g_plus
            r_minus = 1.0 / g_minus

            r_plus = max(R_MIN, min(r_plus, R_MAX))
            r_minus = max(R_MIN, min(r_minus, R_MAX))

            emit(f"R_2p_{i}_{j}  h1_{i}  sum2p_{j}  {fmt_resistance(r_plus)}"
                 f"  $ W2[{i}][{j}]={w:+.4f}")
            emit(f"R_2n_{i}_{j}  h1_{i}  sum2n_{j}  {fmt_resistance(r_minus)}")
            resistor_count += 2

        emit()

        bias_v = b2[j]
        emit(f"B_out_{j}  out_{j}  0  "
             f"V = (i(V_vg2p_{j}) - i(V_vg2n_{j})) * {R_TIA:.1f}"
             f" + {bias_v:.6e}")
        emit()

    # -----------------------------------------------------------------------
    # Section 5: Winner-take-all output
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* OUTPUT STAGE: Winner-Take-All (Analog Comparator Tree)")
    emit(f"* ===========================================================")
    emit()

    # Binary max tree
    if n_out == 1:
        max_expr = "v(out_0)"
    else:
        current_terms = [f"v(out_{j})" for j in range(n_out)]
        while len(current_terms) > 1:
            next_terms = []
            for k in range(0, len(current_terms), 2):
                if k + 1 < len(current_terms):
                    next_terms.append(f"max({current_terms[k]},{current_terms[k+1]})")
                else:
                    next_terms.append(current_terms[k])
            current_terms = next_terms
        max_expr = current_terms[0]

    emit(f"B_max  vmax  0  V = {max_expr}")
    emit()

    for j in range(n_out):
        char_label = ALPHABET[j] if j < len(ALPHABET) else "?"
        emit(f"B_win_{j}  win_{j}  0  "
             f"V = (v(out_{j}) >= v(vmax) - 0.01) ? {V_HIGH} : {V_LOW}"
             f"  $ '{char_label}'")

    emit()

    # -----------------------------------------------------------------------
    # Section 6: Simulation control
    # -----------------------------------------------------------------------
    emit(f"* ===========================================================")
    emit(f"* SIMULATION CONTROL")
    emit(f"* ===========================================================")
    emit()

    emit(f".options reltol=1e-3 abstol=1e-10 vntol=1e-5 gmin=1e-10")
    emit(f".options itl1=500 itl2=200")
    emit()

    emit(f".control")
    emit(f"set filetype=ascii")
    emit()
    emit(f"op")
    emit()

    # Print results
    emit(f"echo \"=== HIDDEN LAYER (after ReLU) ===\"")
    for j in range(n_hidden):
        emit(f"echo \"h1_{j} = $&v(h1_{j})\"")

    emit()
    emit(f"echo \"\"")
    emit(f"echo \"=== OUTPUT LAYER ===\"")
    for j in range(n_out):
        char_label = ALPHABET[j] if j < len(ALPHABET) else "?"
        emit(f"echo \"out_{j} ('{char_label}') = $&v(out_{j})\"")

    emit()
    emit(f"echo \"\"")
    emit(f"echo \"=== PREDICTED CHARACTER ===\"")
    emit(f"echo \"max_output = $&v(vmax)\"")
    for j in range(n_out):
        char_label = ALPHABET[j] if j < len(ALPHABET) else "?"
        emit(f"echo \"win_{j} ('{char_label}') = $&v(win_{j})\"")

    emit()

    # Save data
    h1_vars = " ".join(f"v(h1_{j})" for j in range(n_hidden))
    out_vars = " ".join(f"v(out_{j})" for j in range(n_out))
    win_vars = " ".join(f"v(win_{j})" for j in range(n_out))

    emit(f"wrdata nn_crossbar_hidden.txt {h1_vars}")
    emit(f"wrdata nn_crossbar_output.txt {out_vars}")
    emit(f"wrdata nn_crossbar_winner.txt {win_vars}")

    emit()
    emit(f"echo \"\"")
    emit(f"echo \"Crossbar inference complete for input '{input_text}'.\"")
    emit(f"echo \"Total resistors: {resistor_count}\"")
    emit(f"quit")
    emit(f".endc")
    emit()
    emit(f".end")

    return "\n".join(lines), resistor_count


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def compute_summary(weights: dict, crossbar_resistor_count: int = 0) -> dict:
    """Compute summary statistics about the generated circuit."""
    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]
    n_in = weights["n_in"]
    n_hidden = weights["n_hidden"]
    n_out = weights["n_out"]

    # Count non-zero weights
    w1_nonzero = sum(1 for i in range(n_in) for j in range(n_hidden)
                     if abs(W1[i][j]) > 1e-10)
    w2_nonzero = sum(1 for i in range(n_hidden) for j in range(n_out)
                     if abs(W2[i][j]) > 1e-10)

    # Weight statistics
    w1_flat = [W1[i][j] for i in range(n_in) for j in range(n_hidden)]
    w2_flat = [W2[i][j] for i in range(n_hidden) for j in range(n_out)]
    all_w = w1_flat + w2_flat

    w_max = max(abs(w) for w in all_w) if all_w else 0
    w_mean = sum(abs(w) for w in all_w) / len(all_w) if all_w else 0

    # Node counts
    # Hybrid: in_* (n_in) + h1_raw_* (n_hidden) + h1_* (n_hidden) +
    #         out_* (n_out) + vmax + win_* (n_out)
    hybrid_nodes = n_in + 2 * n_hidden + 2 * n_out + 1

    # Crossbar adds: sum1p_*, sum1n_* (2*n_hidden) + sum2p_*, sum2n_* (2*n_out)
    crossbar_nodes = hybrid_nodes + 2 * n_hidden + 2 * n_out

    # Resistor count for crossbar
    if crossbar_resistor_count == 0:
        crossbar_resistor_count = 2 * (n_in * n_hidden + n_hidden * n_out)

    # Diode count (ReLU): one per hidden neuron
    diode_count = n_hidden

    # Estimated power (rough): P = V^2 / R_avg for active inputs
    # With one-hot encoding, only CONTEXT_LEN inputs are active
    # Power per active input through crossbar: V_HIGH^2 * G_avg * n_columns
    avg_conductance = w_mean * G_SCALE if w_mean > 0 else G_MIN
    est_power_per_input = V_HIGH ** 2 * avg_conductance * (n_hidden + n_out)
    est_total_power = est_power_per_input * CONTEXT_LEN  # only active inputs matter

    return {
        "n_in": n_in,
        "n_hidden": n_hidden,
        "n_out": n_out,
        "total_weights": n_in * n_hidden + n_hidden * n_out,
        "w1_nonzero": w1_nonzero,
        "w2_nonzero": w2_nonzero,
        "w_max": w_max,
        "w_mean": w_mean,
        "hybrid_nodes": hybrid_nodes,
        "crossbar_nodes": crossbar_nodes,
        "crossbar_resistors": crossbar_resistor_count,
        "diodes": diode_count,
        "est_power_mW": est_total_power * 1e3,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate ngspice netlist for analog neural network inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        default="hel",
        help="3-character input string for inference (default: 'hel')",
    )
    parser.add_argument(
        "--batch", "-b",
        default=None,
        help="Comma-separated list of 3-character inputs for batch generation",
    )
    parser.add_argument(
        "--full-crossbar", "-f",
        action="store_true",
        help="Also generate full resistive crossbar netlist",
    )
    parser.add_argument(
        "--weights", "-w",
        default=None,
        help="Path to weights JSON file (default: weights_analog.json in same dir)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory for generated netlists (default: same as script)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress summary output",
    )

    args = parser.parse_args()

    # Resolve paths
    if args.weights:
        weights_path = Path(args.weights).resolve()
    else:
        weights_path = SCRIPT_DIR / "weights.json"

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = SCRIPT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load weights
    if not weights_path.exists():
        print(f"ERROR: Weights file not found: {weights_path}")
        print(f"Run train_model.py first to generate weights_analog.json")
        sys.exit(1)

    if not args.quiet:
        print(f"Loading weights from: {weights_path}")

    weights = load_weights(str(weights_path))

    if not args.quiet:
        print(f"  Architecture: {weights['n_in']} -> {weights['n_hidden']} -> {weights['n_out']}")
        meta = weights.get("metadata", {})
        if meta:
            print(f"  Training accuracy: {meta.get('accuracy', 'N/A')}")
            print(f"  Training loss: {meta.get('loss', 'N/A')}")

    # Determine input(s) to process
    if args.batch:
        inputs = [s.strip() for s in args.batch.split(",")]
    else:
        inputs = [args.input]

    # Validate all inputs
    for text in inputs:
        if len(text) != CONTEXT_LEN:
            print(f"ERROR: Input '{text}' must be exactly {CONTEXT_LEN} characters")
            sys.exit(1)
        for ch in text:
            if ch.lower() not in ALPHABET:
                print(f"ERROR: Character '{ch}' not in alphabet: '{ALPHABET}'")
                sys.exit(1)

    # Generate netlists
    for text in inputs:
        text_lower = text.lower()
        input_vec = encode_input(text_lower)

        # Filename suffix for batch mode
        suffix = f"_{text_lower}" if len(inputs) > 1 else ""

        # --- Hybrid netlist ---
        if not args.quiet:
            print(f"\nGenerating hybrid netlist for input '{text_lower}'...")

        hybrid_netlist = generate_hybrid_netlist(weights, input_vec, text_lower)
        hybrid_path = output_dir / f"analog_nn{suffix}.cir"

        with open(hybrid_path, "w", newline="\n") as f:
            f.write(hybrid_netlist)

        if not args.quiet:
            line_count = len(hybrid_netlist.split("\n"))
            print(f"  Written: {hybrid_path} ({line_count} lines)")

        # --- Full crossbar netlist ---
        if args.full_crossbar:
            if not args.quiet:
                print(f"Generating full crossbar netlist for input '{text_lower}'...")

            crossbar_netlist, r_count = generate_crossbar_netlist(
                weights, input_vec, text_lower
            )
            crossbar_path = output_dir / f"analog_nn_crossbar{suffix}.cir"

            with open(crossbar_path, "w", newline="\n") as f:
                f.write(crossbar_netlist)

            if not args.quiet:
                line_count = len(crossbar_netlist.split("\n"))
                print(f"  Written: {crossbar_path} ({line_count} lines)")

    # --- Summary ---
    if not args.quiet:
        crossbar_r = 0
        if args.full_crossbar:
            # Use the last r_count computed
            crossbar_r = r_count

        summary = compute_summary(weights, crossbar_r)

        print(f"\n{'=' * 60}")
        print(f"  ANALOG NEURAL NETWORK CIRCUIT SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Architecture:     {summary['n_in']} -> {summary['n_hidden']} "
              f"(ReLU) -> {summary['n_out']}")
        print(f"  Total weights:    {summary['total_weights']}")
        print(f"  Non-zero (L1):    {summary['w1_nonzero']}")
        print(f"  Non-zero (L2):    {summary['w2_nonzero']}")
        print(f"  |W| max:          {summary['w_max']:.4f}")
        print(f"  |W| mean:         {summary['w_mean']:.4f}")
        print(f"  Diodes (ReLU):    {summary['diodes']}")
        print(f"  Hybrid nodes:     {summary['hybrid_nodes']}")

        if args.full_crossbar:
            print(f"  Crossbar nodes:   {summary['crossbar_nodes']}")
            print(f"  Crossbar resistors: {summary['crossbar_resistors']}")
            print(f"  Est. power:       {summary['est_power_mW']:.3f} mW")

        print(f"{'=' * 60}")

        # Print input encoding summary
        for text in inputs:
            text_lower = text.lower()
            vec = encode_input(text_lower)
            active = [i for i, v in enumerate(vec) if v > 0]
            print(f"\n  Input '{text_lower}' -> active nodes: {active}")
            for pos, ch in enumerate(text_lower):
                idx = char_to_index(ch)
                global_idx = pos * VOCAB_SIZE + idx
                print(f"    char[{pos}] = '{ch}' -> in_{global_idx} = {V_HIGH}V")


if __name__ == "__main__":
    main()
