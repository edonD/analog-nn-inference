#!/usr/bin/env python3
"""
Generate results.md and progress.svg for GitHub visibility.

Reads experiment_history.json and produces:
  - results.md: markdown table + embedded SVG chart
  - progress.svg: line chart of score over experiments

GitHub renders both natively — no external tools needed.

Usage:
    python update_results.py
"""

import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def load_history():
    path = SCRIPT_DIR / "experiment_history.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {"experiments": [], "best_score": 0, "best_exp_id": None}


def generate_svg(experiments, width=800, height=300):
    """Generate an SVG line chart of score over experiments."""
    if not experiments:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="100"><text x="400" y="50" text-anchor="middle" fill="#888" font-size="14">No experiments yet</text></svg>'

    pad = {"top": 40, "right": 30, "bottom": 50, "left": 60}
    plot_w = width - pad["left"] - pad["right"]
    plot_h = height - pad["top"] - pad["bottom"]

    scores = [e["scores"].get("total_score", 0) for e in experiments]
    analog_accs = [e["scores"].get("analog_accuracy", 0) for e in experiments]
    match_rates = [e["scores"].get("match_rate", 0) for e in experiments]

    n = len(scores)
    max_score = max(max(scores), 1.0)
    min_score = min(min(scores), 0)
    y_range = max_score - min_score if max_score > min_score else 1.0

    def x_pos(i):
        if n == 1:
            return pad["left"] + plot_w / 2
        return pad["left"] + (i / (n - 1)) * plot_w

    def y_pos(v):
        return pad["top"] + plot_h - ((v - min_score) / y_range) * plot_h

    lines = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#0d1117;border-radius:8px;">')

    # Title
    lines.append(f'<text x="{width/2}" y="22" text-anchor="middle" fill="#e0e0e0" font-size="14" font-weight="bold">Analog MNIST Autoresearch — Score Progression</text>')

    # Grid lines
    for i in range(5):
        val = min_score + y_range * i / 4
        y = y_pos(val)
        lines.append(f'<line x1="{pad["left"]}" y1="{y}" x2="{width-pad["right"]}" y2="{y}" stroke="#1a2030" stroke-width="1"/>')
        lines.append(f'<text x="{pad["left"]-8}" y="{y+4}" text-anchor="end" fill="#666" font-size="11" font-family="monospace">{val:.2f}</text>')

    # X-axis labels
    step = max(1, n // 10)
    for i in range(0, n, step):
        x = x_pos(i)
        lines.append(f'<text x="{x}" y="{height-pad["bottom"]+20}" text-anchor="middle" fill="#666" font-size="10">#{experiments[i]["id"]}</text>')

    # Axis labels
    lines.append(f'<text x="{width/2}" y="{height-8}" text-anchor="middle" fill="#888" font-size="11">Experiment</text>')
    lines.append(f'<text x="14" y="{height/2}" text-anchor="middle" fill="#888" font-size="11" transform="rotate(-90,14,{height/2})">Score</text>')

    # Plot lines
    def draw_line(values, color, label, dash=""):
        if len(values) < 2:
            x = x_pos(0)
            y = y_pos(values[0])
            lines.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')
            return
        points = " ".join(f"{x_pos(i)},{y_pos(v)}" for i, v in enumerate(values))
        stroke_dash = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"{stroke_dash}/>')

        # Points
        for i, v in enumerate(values):
            is_best = experiments[i].get("is_best", False)
            r = 5 if is_best else 3
            fill = "#ffcc00" if is_best else color
            lines.append(f'<circle cx="{x_pos(i)}" cy="{y_pos(v)}" r="{r}" fill="{fill}"/>')
            if is_best:
                lines.append(f'<text x="{x_pos(i)}" y="{y_pos(v)-10}" text-anchor="middle" fill="#ffcc00" font-size="10" font-weight="bold">{v:.3f}</text>')

    draw_line(scores, "#00ff88", "Total Score")
    draw_line(analog_accs, "#4488ff", "Analog Acc", "5,3")
    draw_line(match_rates, "#ff8844", "Match Rate", "3,3")

    # Legend
    legend_x = pad["left"] + 10
    legend_y = pad["top"] + 10
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="160" height="60" rx="4" fill="#0d1117" stroke="#333"/>')
    for i, (color, label, dash) in enumerate([("#00ff88", "Total Score", ""), ("#4488ff", "Analog Accuracy", "5,3"), ("#ff8844", "Match Rate", "3,3")]):
        y = legend_y + 15 + i * 18
        d = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(f'<line x1="{legend_x+8}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{color}" stroke-width="2"{d}/>')
        lines.append(f'<text x="{legend_x+34}" y="{y+4}" fill="#ccc" font-size="10">{label}</text>')

    # Target line at 0.80
    if min_score < 0.8 < max_score:
        y80 = y_pos(0.8)
        lines.append(f'<line x1="{pad["left"]}" y1="{y80}" x2="{width-pad["right"]}" y2="{y80}" stroke="#ff4444" stroke-width="1" stroke-dasharray="8,4"/>')
        lines.append(f'<text x="{width-pad["right"]+5}" y="{y80+4}" fill="#ff4444" font-size="10">target</text>')

    lines.append('</svg>')
    return "\n".join(lines)


def generate_markdown(history):
    """Generate results.md with table and embedded chart."""
    experiments = history.get("experiments", [])
    best_score = history.get("best_score", 0)
    best_id = history.get("best_exp_id", None)

    lines = []
    lines.append("# Analog MNIST — Autoresearch Results")
    lines.append("")
    lines.append("> A resistive crossbar circuit that classifies handwritten digits through pure analog physics.")
    lines.append("> No CPU. No GPU. No code execution during inference. Just Ohm's law and Kirchhoff's law.")
    lines.append("")

    # Stats
    if experiments:
        latest = experiments[-1]
        latest_score = latest["scores"].get("total_score", 0)
        latest_acc = latest["scores"].get("analog_accuracy", 0)
        lines.append(f"**Best Score: {best_score:.4f}** (experiment #{best_id})")
        lines.append(f"| Latest Score: {latest_score:.4f} | Experiments: {len(experiments)} |")
        lines.append("")

    # Chart
    lines.append("## Score Progression")
    lines.append("")
    lines.append("![Progress](progress.svg)")
    lines.append("")

    # Challenge description
    lines.append("## The Challenge")
    lines.append("")
    lines.append("| Constraint | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Input | 14×14 MNIST digits (196 pixels) |")
    lines.append("| Architecture | 196 → 64 (ReLU) → 10 |")
    lines.append("| Resistors | ~26,000 (differential crossbar) |")
    lines.append("| Diode model | **Realistic** (n=1.0, Is=1e-14) — 0.7V forward drop |")
    lines.append("| Resistor mismatch | **5% random variation** on every resistor |")
    lines.append("| Target | ≥ 80% analog accuracy on 100 test digits |")
    lines.append("")

    # Experiment table
    if experiments:
        lines.append("## Experiment History")
        lines.append("")
        lines.append("| # | Score | Analog Acc | Match | CosSim | Notes |")
        lines.append("|---|-------|-----------|-------|--------|-------|")

        for exp in experiments:
            s = exp["scores"]
            best_marker = " ⭐" if exp.get("is_best") else ""
            score = s.get("total_score", 0)
            acc = s.get("analog_accuracy", 0)
            match = s.get("match_rate", 0)
            cos = s.get("avg_cosine_sim", 0)
            notes = exp.get("notes", "")
            lines.append(f"| {exp['id']} | **{score:.4f}**{best_marker} | {acc:.1%} | {match:.1%} | {cos:.4f} | {notes} |")

        lines.append("")

    # Best params
    if best_id and experiments:
        best_exp = next((e for e in experiments if e["id"] == best_id), None)
        if best_exp and "params" in best_exp:
            lines.append("## Best Parameters")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(best_exp["params"], indent=2))
            lines.append("```")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Auto-generated by `update_results.py`. Each experiment runs real ngspice simulations.*")
    lines.append("")

    return "\n".join(lines)


def main():
    history = load_history()
    experiments = history.get("experiments", [])

    # Generate SVG
    svg = generate_svg(experiments)
    svg_path = SCRIPT_DIR / "progress.svg"
    with open(svg_path, "w") as f:
        f.write(svg)
    print(f"Saved: {svg_path}")

    # Generate markdown
    md = generate_markdown(history)
    md_path = SCRIPT_DIR / "results.md"
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
