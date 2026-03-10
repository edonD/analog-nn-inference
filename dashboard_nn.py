#!/usr/bin/env python3
"""
Interactive Plotly HTML dashboard that visualizes the analog neural network
inference pipeline.

Generates a self-contained nn_dashboard.html with 6 panels (3 rows x 2 cols):
    Row 1: Network architecture diagram | Hidden layer activations
    Row 2: Output probabilities (digital vs analog) | Error analysis scatter
    Row 3: Batch results table | Weight heatmaps

Usage:
    python dashboard_nn.py                    # Generate dashboard
    python dashboard_nn.py --input "hel"     # Focus on specific input
    python dashboard_nn.py --open            # Generate and open in browser
"""

import argparse
import json
import os
import sys
import webbrowser
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    print("ERROR: plotly is required.  Install with:  pip install plotly")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Constants (must match train_model.py)
# ---------------------------------------------------------------------------
VOCAB = {" ": 0}
for _i, _ch in enumerate("abcdefghijklmnopqrstuvwxyz", start=1):
    VOCAB[_ch] = _i
INV_VOCAB = {v: k for k, v in VOCAB.items()}

VOCAB_SIZE = 27
CONTEXT_LEN = 3
INPUT_DIM = CONTEXT_LEN * VOCAB_SIZE  # 81
HIDDEN_DIM = 32
OUTPUT_DIM = VOCAB_SIZE  # 27

CHAR_LABELS = [INV_VOCAB[i] if INV_VOCAB[i] != " " else "' '" for i in range(VOCAB_SIZE)]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_inference_results(filepath=None):
    """Load inference_results.json, or run inference to create it."""
    if filepath is None:
        filepath = os.path.join(SCRIPT_DIR, "inference_results.json")

    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            return json.load(f)

    # Results file does not exist -- run inference to generate it
    print("inference_results.json not found, running inference...")
    try:
        # Import run_inference from the same directory
        sys.path.insert(0, SCRIPT_DIR)
        import run_inference

        results = run_inference.run_all_inferences(
            run_inference.TEST_INPUTS,
            digital_only=True,  # default to digital-only if running standalone
            verbose=True,
        )
        run_inference.save_results(results, filepath)

        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as exc:
        print(f"Could not run inference: {exc}")
        sys.exit(1)


def load_weights_direct():
    """Load weights.json directly."""
    path = os.path.join(SCRIPT_DIR, "weights.json")
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Inline inference (for focused-input mode)
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def softmax_1d(logits):
    shifted = logits - logits.max()
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum()


def encode_input(input_text):
    text = input_text.lower()
    text = "".join(ch if ch in VOCAB else " " for ch in text)
    if len(text) < CONTEXT_LEN:
        text = " " * (CONTEXT_LEN - len(text)) + text
    ctx = text[-CONTEXT_LEN:]
    x = np.zeros(INPUT_DIM, dtype=np.float64)
    for pos in range(CONTEXT_LEN):
        x[pos * VOCAB_SIZE + VOCAB[ctx[pos]]] = 1.0
    return x


def run_single_inference(input_text, W1, b1, W2, b2):
    """Run digital inference for a single input, returning all intermediates."""
    x = encode_input(input_text)
    z1 = x @ W1 + b1
    hidden = relu(z1)
    logits = hidden @ W2 + b2
    probs = softmax_1d(logits)
    return {
        "input": input_text,
        "input_vec": x,
        "hidden": hidden,
        "logits": logits,
        "probs": probs,
        "pred_idx": int(np.argmax(probs)),
    }


# ---------------------------------------------------------------------------
# Panel 1: Network Architecture Diagram
# ---------------------------------------------------------------------------
def make_architecture_panel(focus_data):
    """
    Create a network architecture scatter plot with lines representing
    connections. Nodes are colored by their activation values.
    """
    fig = go.Figure()

    input_vec = np.array(focus_data.get("digital_input", focus_data.get("input_vec", [0] * INPUT_DIM)))
    hidden_vec = np.array(focus_data.get("digital_hidden", focus_data.get("hidden", [0] * HIDDEN_DIM)))
    probs_vec = np.array(focus_data.get("digital_probs", focus_data.get("probs", [0] * OUTPUT_DIM)))

    # Layout: 3 columns at x = 0, 1, 2
    # Subsample input nodes for readability (show 27 per position slot)
    # We'll show all 81 but group them visually

    n_input = INPUT_DIM     # 81
    n_hidden = HIDDEN_DIM   # 32
    n_output = OUTPUT_DIM   # 27

    # Y positions -- spread nodes vertically
    def y_positions(n, spread=1.0):
        return np.linspace(-spread / 2, spread / 2, n)

    input_y = y_positions(n_input, 10.0)
    hidden_y = y_positions(n_hidden, 8.0)
    output_y = y_positions(n_output, 6.0)

    input_x = np.zeros(n_input)
    hidden_x = np.ones(n_hidden)
    output_x = np.ones(n_output) * 2

    # Draw connections (subsample to avoid visual clutter)
    # Layer 1: show connections from active inputs to hidden
    active_inputs = np.where(input_vec > 0.5)[0]  # one-hot, so these are the 3 active ones
    for inp_idx in active_inputs:
        for hid_idx in range(n_hidden):
            if abs(hidden_vec[hid_idx]) > 0.01:  # only show connections to firing neurons
                opacity = min(0.6, abs(hidden_vec[hid_idx]) / (abs(hidden_vec).max() + 1e-9))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[input_y[inp_idx], hidden_y[hid_idx]],
                    mode="lines",
                    line=dict(color=f"rgba(100,149,237,{opacity:.4f})", width=0.5),
                    hoverinfo="skip", showlegend=False,
                ))

    # Layer 2: show connections from active hidden to top-5 outputs
    top5_outputs = np.argsort(probs_vec)[::-1][:5]
    active_hidden = np.where(hidden_vec > 0.01)[0]
    for hid_idx in active_hidden:
        for out_idx in top5_outputs:
            opacity = min(0.6, float(probs_vec[out_idx]))
            fig.add_trace(go.Scatter(
                x=[1, 2], y=[hidden_y[hid_idx], output_y[out_idx]],
                mode="lines",
                line=dict(color=f"rgba(255,165,0,{opacity:.4f})", width=0.5),
                hoverinfo="skip", showlegend=False,
            ))

    # Input layer labels
    input_labels = []
    for i in range(n_input):
        pos = i // VOCAB_SIZE
        char_idx = i % VOCAB_SIZE
        ch = INV_VOCAB[char_idx]
        ch_display = "SPC" if ch == " " else ch
        input_labels.append(f"pos{pos}:{ch_display}")

    # Draw input nodes
    fig.add_trace(go.Scatter(
        x=input_x, y=input_y,
        mode="markers",
        marker=dict(
            size=4,
            color=input_vec,
            colorscale=[[0, "#1a1a2e"], [1, "#00ff88"]],
            cmin=0, cmax=1,
            line=dict(width=0.3, color="#444"),
        ),
        text=input_labels, hoverinfo="text",
        name="Input (81)", showlegend=True,
    ))

    # Draw hidden nodes
    hidden_labels = [f"h{i}: {hidden_vec[i]:.3f}" for i in range(n_hidden)]
    h_max = abs(hidden_vec).max() if abs(hidden_vec).max() > 0 else 1.0
    fig.add_trace(go.Scatter(
        x=hidden_x, y=hidden_y,
        mode="markers",
        marker=dict(
            size=8,
            color=hidden_vec,
            colorscale="YlOrRd",
            cmin=0, cmax=h_max,
            line=dict(width=0.5, color="#444"),
        ),
        text=hidden_labels, hoverinfo="text",
        name="Hidden (32)", showlegend=True,
    ))

    # Draw output nodes
    output_labels = [f"{CHAR_LABELS[i]}: {probs_vec[i]:.3f}" for i in range(n_output)]
    fig.add_trace(go.Scatter(
        x=output_x, y=output_y,
        mode="markers+text",
        marker=dict(
            size=10,
            color=probs_vec,
            colorscale="Blues",
            cmin=0, cmax=max(probs_vec.max(), 0.01),
            line=dict(width=0.5, color="#444"),
        ),
        text=CHAR_LABELS,
        textposition="middle right",
        textfont=dict(size=8),
        hovertext=output_labels, hoverinfo="text",
        name="Output (27)", showlegend=True,
    ))

    # Column labels
    for x_pos, label in [(0, "Input (81)"), (1, "Hidden (32)"), (2, "Output (27)")]:
        fig.add_annotation(
            x=x_pos, y=6.0, text=f"<b>{label}</b>",
            showarrow=False, font=dict(size=12),
        )

    input_text = focus_data.get("input", "???")
    fig.update_layout(
        title=dict(text=f"Network Architecture -- input: \"{input_text}\"", font=dict(size=14)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.3, 2.7]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#fafafa",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


# ---------------------------------------------------------------------------
# Panel 2: Hidden Layer Activations
# ---------------------------------------------------------------------------
def make_hidden_activations_panel(focus_data):
    """Bar chart of all 32 hidden neuron activations."""
    hidden = np.array(focus_data.get("digital_hidden", focus_data.get("hidden", [0] * HIDDEN_DIM)))
    neuron_ids = [f"h{i}" for i in range(HIDDEN_DIM)]

    colors = ["#d32f2f" if v > 0.01 else "#bdbdbd" for v in hidden]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=neuron_ids,
        y=hidden,
        marker_color=colors,
        hovertemplate="Neuron %{x}<br>Activation: %{y:.4f}<extra></extra>",
    ))

    n_active = int(np.sum(hidden > 0.01))
    n_dead = HIDDEN_DIM - n_active

    fig.update_layout(
        title=dict(
            text=f"Hidden Layer Activations -- {n_active} active, {n_dead} dead (ReLU=0)",
            font=dict(size=14),
        ),
        xaxis_title="Neuron",
        yaxis_title="Activation (post-ReLU)",
        plot_bgcolor="#fafafa",
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Panel 3: Output Probabilities -- Digital vs Analog
# ---------------------------------------------------------------------------
def make_output_comparison_panel(focus_data):
    """Grouped bar chart of digital probs vs analog voltages."""
    digital_probs = np.array(focus_data.get("digital_probs", focus_data.get("probs", [0] * OUTPUT_DIM)))

    has_analog = focus_data.get("analog_available", False)
    analog_outputs = np.array(focus_data.get("analog_outputs", [0] * OUTPUT_DIM)) if has_analog else None

    # Normalize analog outputs to [0,1] for comparison if available
    analog_norm = None
    if analog_outputs is not None:
        a_min = analog_outputs.min()
        a_max = analog_outputs.max()
        if a_max - a_min > 1e-12:
            analog_norm = (analog_outputs - a_min) / (a_max - a_min)
        else:
            analog_norm = analog_outputs.copy()

    digital_pred = int(np.argmax(digital_probs))

    fig = go.Figure()

    # Digital bars
    digital_colors = ["#1565c0" if i != digital_pred else "#0d47a1" for i in range(OUTPUT_DIM)]
    fig.add_trace(go.Bar(
        x=CHAR_LABELS,
        y=digital_probs,
        name="Digital (softmax)",
        marker_color=digital_colors,
        hovertemplate="%{x}: %{y:.4f}<extra>Digital</extra>",
    ))

    # Analog bars
    if analog_norm is not None:
        analog_pred = int(np.argmax(analog_outputs))
        analog_colors = ["#e65100" if i != analog_pred else "#bf360c" for i in range(OUTPUT_DIM)]
        fig.add_trace(go.Bar(
            x=CHAR_LABELS,
            y=analog_norm,
            name="Analog (normalized)",
            marker_color=analog_colors,
            hovertemplate="%{x}: %{y:.4f}<extra>Analog (norm)</extra>",
        ))

    # Highlight predicted character
    fig.add_annotation(
        x=CHAR_LABELS[digital_pred],
        y=digital_probs[digital_pred],
        text=f"pred: {CHAR_LABELS[digital_pred]}",
        showarrow=True, arrowhead=2,
        font=dict(size=10, color="#0d47a1"),
    )

    title_suffix = "(digital only)" if not has_analog else "(digital vs analog)"
    fig.update_layout(
        title=dict(text=f"Output Probabilities {title_suffix}", font=dict(size=14)),
        xaxis_title="Character",
        yaxis_title="Probability / Normalized Voltage",
        barmode="group",
        plot_bgcolor="#fafafa",
        margin=dict(l=50, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


# ---------------------------------------------------------------------------
# Panel 4: Error Analysis Scatter
# ---------------------------------------------------------------------------
def make_error_analysis_panel(focus_data):
    """
    Scatter plot of digital vs analog output values.
    If analog is not available, show a digital self-correlation plot
    (probs vs logits-based ranking) with a note.
    """
    digital_probs = np.array(focus_data.get("digital_probs", focus_data.get("probs", [0] * OUTPUT_DIM)))
    has_analog = focus_data.get("analog_available", False)

    fig = go.Figure()

    if has_analog:
        analog_outputs = np.array(focus_data["analog_outputs"])
        # Normalize analog to same scale as digital probs
        a_min = analog_outputs.min()
        a_max = analog_outputs.max()
        if a_max - a_min > 1e-12:
            analog_norm = (analog_outputs - a_min) / (a_max - a_min)
        else:
            analog_norm = analog_outputs.copy()

        # R-squared
        ss_res = np.sum((digital_probs - analog_norm) ** 2)
        ss_tot = np.sum((digital_probs - digital_probs.mean()) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-12)
        max_err = float(np.max(np.abs(digital_probs - analog_norm)))

        fig.add_trace(go.Scatter(
            x=digital_probs, y=analog_norm,
            mode="markers+text",
            text=CHAR_LABELS,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(size=8, color="#e65100"),
            hovertemplate="Digital: %{x:.4f}<br>Analog: %{y:.4f}<br>%{text}<extra></extra>",
            name="Outputs",
        ))

        # Perfect correlation diagonal
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(color="gray", dash="dash", width=1),
            showlegend=False, hoverinfo="skip",
        ))

        fig.update_layout(
            title=dict(text=f"Error Analysis -- R²={r_squared:.4f}, Max err={max_err:.4f}", font=dict(size=14)),
            xaxis_title="Digital (softmax prob)",
            yaxis_title="Analog (normalized voltage)",
        )
    else:
        # No analog data -- show digital probabilities ranked
        sorted_indices = np.argsort(digital_probs)[::-1]
        ranks = np.arange(1, OUTPUT_DIM + 1)
        sorted_probs = digital_probs[sorted_indices]
        sorted_labels = [CHAR_LABELS[i] for i in sorted_indices]

        fig.add_trace(go.Scatter(
            x=ranks, y=sorted_probs,
            mode="markers+text+lines",
            text=sorted_labels,
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=8, color=sorted_probs, colorscale="Blues", cmin=0, cmax=max(sorted_probs.max(), 0.01)),
            line=dict(color="#90caf9", width=1),
            hovertemplate="Rank %{x}<br>Prob: %{y:.4f}<br>%{text}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text="Digital Output Ranking (analog not available)", font=dict(size=14)),
            xaxis_title="Rank",
            yaxis_title="Softmax Probability",
        )
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="Analog data not available -- run with ngspice for comparison",
            showarrow=False, font=dict(size=11, color="gray"),
        )

    fig.update_layout(
        plot_bgcolor="#fafafa",
        margin=dict(l=50, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Panel 5: Batch Results Table
# ---------------------------------------------------------------------------
def make_results_table(all_results):
    """Table showing all test inputs with predictions and metrics."""
    inputs = []
    d_preds = []
    d_probs = []
    a_preds = []
    matches = []
    cos_sims = []
    max_errs = []

    for r in all_results:
        inputs.append(r["input"])
        d_preds.append(r.get("digital_pred_char", "?"))
        d_probs.append(f"{r.get('digital_pred_prob', 0):.3f}")

        if r.get("analog_available", False):
            a_preds.append(r.get("analog_pred_char", "?"))
            matches.append("YES" if r.get("match") else "NO")
            cos_sims.append(f"{r.get('cosine_sim', 0):.3f}")
            max_errs.append(f"{r.get('max_error', 0):.4f}")
        else:
            a_preds.append("--")
            matches.append("--")
            cos_sims.append("--")
            max_errs.append("--")

    # Color cells for match column
    match_colors = []
    for m in matches:
        if m == "YES":
            match_colors.append("#c8e6c9")
        elif m == "NO":
            match_colors.append("#ffcdd2")
        else:
            match_colors.append("#f5f5f5")

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Input</b>", "<b>Digital Pred</b>", "<b>Prob</b>",
                    "<b>Analog Pred</b>", "<b>Match</b>", "<b>Cos Sim</b>", "<b>Max Err</b>"],
            fill_color="#1565c0",
            font=dict(color="white", size=11),
            align="center",
            height=30,
        ),
        cells=dict(
            values=[inputs, d_preds, d_probs, a_preds, matches, cos_sims, max_errs],
            fill_color=[
                ["#fafafa"] * len(inputs),  # input
                ["#e3f2fd"] * len(inputs),  # digital pred
                ["#fafafa"] * len(inputs),  # prob
                ["#fff3e0"] * len(inputs),  # analog pred
                match_colors,               # match
                ["#fafafa"] * len(inputs),  # cos sim
                ["#fafafa"] * len(inputs),  # max err
            ],
            align="center",
            height=25,
            font=dict(size=11),
        ),
    )])

    fig.update_layout(
        title=dict(text=f"Batch Results -- {len(all_results)} test cases", font=dict(size=14)),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# Panel 6: Weight Heatmaps
# ---------------------------------------------------------------------------
def make_weight_heatmaps(weights_data):
    """Two heatmaps: Layer 1 (81x32) and Layer 2 (32x27)."""
    from plotly.subplots import make_subplots

    W1 = np.array(weights_data["W1"])  # 81 x 32
    W2 = np.array(weights_data["W2"])  # 32 x 27

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Layer 1 Weights (81x32)", "Layer 2 Weights (32x27)"],
        horizontal_spacing=0.08,
    )

    # Layer 1 heatmap
    # Y-axis: input features (group by position)
    w1_labels = []
    for pos in range(CONTEXT_LEN):
        for ci in range(VOCAB_SIZE):
            ch = INV_VOCAB[ci]
            ch_d = "SPC" if ch == " " else ch
            w1_labels.append(f"p{pos}:{ch_d}")

    fig.add_trace(go.Heatmap(
        z=W1,
        x=[f"h{i}" for i in range(HIDDEN_DIM)],
        y=w1_labels,
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="W1", x=0.44, len=0.9),
        hovertemplate="Input: %{y}<br>Hidden: %{x}<br>Weight: %{z:.4f}<extra></extra>",
    ), row=1, col=1)

    # Layer 2 heatmap
    fig.add_trace(go.Heatmap(
        z=W2,
        x=CHAR_LABELS,
        y=[f"h{i}" for i in range(HIDDEN_DIM)],
        colorscale="RdBu_r",
        zmid=0,
        colorbar=dict(title="W2", x=1.0, len=0.9),
        hovertemplate="Hidden: %{y}<br>Output: %{x}<br>Weight: %{z:.4f}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text="Weight Heatmaps", font=dict(size=14)),
        margin=dict(l=80, r=20, t=50, b=40),
        height=600,
    )
    return fig


# ---------------------------------------------------------------------------
# Assemble full dashboard
# ---------------------------------------------------------------------------
def build_dashboard(all_results, weights_data, focus_input=None):
    """
    Build the complete 6-panel dashboard as a single HTML string.

    We create each panel as an independent Plotly figure and combine them
    into a single HTML page using div blocks.
    """
    # Determine which result to focus on for panels 1-4
    if focus_input:
        focus_data = None
        for r in all_results:
            if r["input"] == focus_input:
                focus_data = r
                break
        if focus_data is None:
            # Run inference for this input on the fly
            W1 = np.array(weights_data["W1"])
            b1 = np.array(weights_data["b1"])
            W2 = np.array(weights_data["W2"])
            b2 = np.array(weights_data["b2"])
            inf = run_single_inference(focus_input, W1, b1, W2, b2)
            focus_data = {
                "input": focus_input,
                "digital_input": inf["input_vec"].tolist(),
                "digital_hidden": inf["hidden"].tolist(),
                "digital_probs": inf["probs"].tolist(),
                "digital_pred_idx": inf["pred_idx"],
                "digital_pred_char": INV_VOCAB[inf["pred_idx"]],
                "digital_pred_prob": float(inf["probs"][inf["pred_idx"]]),
                "analog_available": False,
            }
    else:
        # Default: use the first result
        focus_data = all_results[0] if all_results else None

    if focus_data is None:
        print("ERROR: No results to visualize.")
        sys.exit(1)

    # Build each panel
    fig1 = make_architecture_panel(focus_data)
    fig2 = make_hidden_activations_panel(focus_data)
    fig3 = make_output_comparison_panel(focus_data)
    fig4 = make_error_analysis_panel(focus_data)
    fig5 = make_results_table(all_results)
    fig6 = make_weight_heatmaps(weights_data)

    # Set consistent sizing
    panel_height = 500
    for fig in [fig1, fig2, fig3, fig4]:
        fig.update_layout(height=panel_height)
    fig5.update_layout(height=min(400, 80 + 30 * len(all_results)))
    fig6.update_layout(height=600)

    # Generate HTML for each panel
    import plotly.io as pio

    html_parts = []
    panel_configs = [
        (fig1, "panel1", "Network Architecture"),
        (fig2, "panel2", "Hidden Activations"),
        (fig3, "panel3", "Output Probabilities"),
        (fig4, "panel4", "Error Analysis"),
        (fig5, "panel5", "Batch Results"),
        (fig6, "panel6", "Weight Heatmaps"),
    ]

    for fig, div_id, _ in panel_configs:
        html_parts.append(
            pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=div_id)
        )

    # Compose the full page
    focus_label = focus_data["input"]
    title = f"Analog Neural Network Dashboard -- focus: \"{focus_label}\""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f0f2f5;
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #1a237e, #0d47a1);
            color: white;
            padding: 20px 30px;
            text-align: center;
        }}
        .header h1 {{ font-size: 1.6em; margin-bottom: 5px; }}
        .header p {{ font-size: 0.95em; opacity: 0.85; }}
        .dashboard {{
            max-width: 1600px;
            margin: 20px auto;
            padding: 0 15px;
        }}
        .row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 15px;
        }}
        .row.full {{
            grid-template-columns: 1fr;
        }}
        .panel {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            padding: 10px;
            overflow: hidden;
        }}
        .footer {{
            text-align: center;
            padding: 15px;
            color: #888;
            font-size: 0.85em;
        }}
        @media (max-width: 900px) {{
            .row {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Analog Neural Network Dashboard</h1>
        <p>Character-level next-char prediction &mdash; 81 inputs &rarr; 32 hidden (ReLU) &rarr; 27 outputs (softmax)</p>
        <p>Focused input: <strong>"{focus_label}"</strong></p>
    </div>

    <div class="dashboard">
        <!-- Row 1: Architecture + Hidden Activations -->
        <div class="row">
            <div class="panel">{html_parts[0]}</div>
            <div class="panel">{html_parts[1]}</div>
        </div>

        <!-- Row 2: Output Comparison + Error Analysis -->
        <div class="row">
            <div class="panel">{html_parts[2]}</div>
            <div class="panel">{html_parts[3]}</div>
        </div>

        <!-- Row 3: Batch Results Table -->
        <div class="row full">
            <div class="panel">{html_parts[4]}</div>
        </div>

        <!-- Row 4: Weight Heatmaps -->
        <div class="row full">
            <div class="panel">{html_parts[5]}</div>
        </div>
    </div>

    <div class="footer">
        Generated by dashboard_nn.py &mdash; Analog Neural Network Inference Visualization
    </div>
</body>
</html>"""

    return html


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate an interactive Plotly dashboard for analog neural network inference.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help='Focus on a specific 3-character input (e.g. "hel")',
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the dashboard in the default browser after generating",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to inference_results.json (default: auto-detect in script dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path (default: nn_dashboard.html in script dir)",
    )
    args = parser.parse_args()

    # Load data
    data = load_inference_results(args.results)
    all_results = data["results"]
    weights_data = data["weights"]

    # Build dashboard
    html = build_dashboard(all_results, weights_data, focus_input=args.input)

    # Write HTML
    out_path = args.output or os.path.join(SCRIPT_DIR, "nn_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Dashboard saved to {out_path}")

    if args.open_browser:
        url = "file://" + os.path.abspath(out_path).replace("\\", "/")
        print(f"Opening {url}")
        webbrowser.open(url)


if __name__ == "__main__":
    main()
