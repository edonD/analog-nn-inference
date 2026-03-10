#!/usr/bin/env python3
"""
Character-level neural network for next-character prediction.

Architecture:
    Input:  3 one-hot vectors concatenated (81 values)
    Layer1: Linear 81->32, ReLU
    Layer2: Linear 32->27, Softmax
    Output: probability distribution over 27 characters (a-z + space)

Pure numpy implementation with manual backprop and Adam optimizer.
Exports weights for analog crossbar circuit implementation.

Usage:
    python train_model.py          # train + export + demo
    python train_model.py --demo   # just run demo with existing weights
"""

import argparse
import json
import os
import sys
import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------
VOCAB = {" ": 0}
for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz", start=1):
    VOCAB[ch] = i
INV_VOCAB = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = 27
CONTEXT_LEN = 3
INPUT_DIM = CONTEXT_LEN * VOCAB_SIZE   # 81
HIDDEN_DIM = 32
OUTPUT_DIM = VOCAB_SIZE                 # 27

# ---------------------------------------------------------------------------
# Training corpus -- ~200 common English words
# ---------------------------------------------------------------------------
WORDS = [
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "his", "how", "its", "may",
    "new", "now", "old", "see", "two", "way", "who", "did", "get", "let",
    "say", "she", "too", "use", "man", "day", "any", "few", "big", "end",
    "hello", "world", "computer", "neural", "network", "analog", "digital",
    "circuit", "voltage", "current", "resistor", "transistor", "silicon",
    "chip", "design", "power", "signal", "frequency", "amplifier", "diode",
    "capacitor", "inductor", "ground", "source", "drain", "gate", "oxide",
    "layer", "wafer", "process", "electron", "photon", "energy", "field",
    "charge", "device", "model", "array", "memory", "logic", "clock",
    "phase", "noise", "gain", "bandwidth", "filter", "output", "input",
    "bias", "threshold", "channel", "drift", "diffusion", "junction",
    "about", "after", "again", "being", "could", "every", "first", "found",
    "great", "house", "large", "learn", "never", "other", "place", "plant",
    "point", "right", "small", "sound", "spell", "still", "study", "their",
    "there", "these", "thing", "think", "three", "water", "where", "which",
    "world", "would", "write", "young", "above", "along", "began", "below",
    "black", "bring", "brown", "build", "carry", "catch", "cause", "clean",
    "close", "color", "cover", "cross", "dance", "don't", "drawn", "drink",
    "drive", "earth", "eight", "equal", "exact", "extra", "final", "floor",
    "force", "front", "green", "group", "heart", "heavy", "horse", "human",
    "image", "known", "laugh", "least", "leave", "level", "light", "match",
    "metal", "might", "month", "mouth", "music", "night", "north", "noted",
    "offer", "often", "order", "paper", "party", "piece", "plain", "press",
    "price", "prove", "queen", "quick", "quite", "radio", "raise", "range",
    "reach", "ready", "river", "scene", "sense", "serve", "seven", "shape",
    "share", "sharp", "short", "shown", "sight", "since", "sleep", "smile",
    "space", "speak", "speed", "spend", "spoke", "stage", "stand", "start",
    "state", "steam", "steel", "stone", "store", "story", "sugar", "table",
    "taken", "teach", "throw", "today", "total", "touch", "train", "tried",
    "truck", "truth", "under", "until", "value", "voice", "watch", "wheel",
    "white", "whole", "women",
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def build_dataset(words):
    """Extract (3-char context -> next char) pairs, padding word starts."""
    contexts = []
    targets = []
    for word in words:
        # Lowercase and filter to valid chars only
        word = word.lower()
        word = "".join(ch for ch in word if ch in VOCAB)
        if len(word) == 0:
            continue
        # Pad the beginning with spaces so every character has a 3-char context
        padded = " " * CONTEXT_LEN + word
        for i in range(len(word)):
            ctx = padded[i : i + CONTEXT_LEN]
            nxt = padded[i + CONTEXT_LEN]
            ctx_indices = [VOCAB[c] for c in ctx]
            tgt_index = VOCAB[nxt]
            contexts.append(ctx_indices)
            targets.append(tgt_index)
    return np.array(contexts, dtype=np.int32), np.array(targets, dtype=np.int32)


def one_hot_encode(contexts):
    """Convert context indices to concatenated one-hot vectors (N, 81)."""
    N = contexts.shape[0]
    X = np.zeros((N, INPUT_DIM), dtype=np.float64)
    for i in range(N):
        for pos in range(CONTEXT_LEN):
            X[i, pos * VOCAB_SIZE + contexts[i, pos]] = 1.0
    return X


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(z.dtype)


def softmax(logits):
    # Numerically stable softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, targets):
    N = probs.shape[0]
    log_probs = -np.log(probs[np.arange(N), targets] + 1e-12)
    return log_probs.mean()


# ---------------------------------------------------------------------------
# Adam optimizer (manual numpy implementation)
# ---------------------------------------------------------------------------
class AdamOptimizer:
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        # First and second moment estimates for each parameter
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params, grads):
        self.t += 1
        updated = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p_new = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            updated.append(p_new)
        return updated


# ---------------------------------------------------------------------------
# Neural network: forward / backward
# ---------------------------------------------------------------------------
def forward(X, W1, b1, W2, b2):
    """Full forward pass. Returns output probs and cached intermediates."""
    z1 = X @ W1 + b1           # (N, 32)
    a1 = relu(z1)              # (N, 32)
    z2 = a1 @ W2 + b2          # (N, 27)
    probs = softmax(z2)        # (N, 27)
    cache = (X, z1, a1, z2)
    return probs, cache


def backward(probs, targets, cache, W1, b1, W2, b2):
    """Backprop through cross-entropy + softmax + relu + linear layers."""
    X, z1, a1, z2 = cache
    N = X.shape[0]

    # Gradient of loss w.r.t. z2 (softmax + cross-entropy shortcut)
    dz2 = probs.copy()
    dz2[np.arange(N), targets] -= 1.0
    dz2 /= N

    # Gradients for W2, b2
    dW2 = a1.T @ dz2           # (32, 27)
    db2 = dz2.sum(axis=0)      # (27,)

    # Backprop into hidden layer
    da1 = dz2 @ W2.T           # (N, 32)
    dz1 = da1 * relu_deriv(z1) # (N, 32)

    # Gradients for W1, b1
    dW1 = X.T @ dz1            # (81, 32)
    db1 = dz1.sum(axis=0)      # (32,)

    return dW1, db1, dW2, db2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(epochs=1000, lr=0.01):
    """Train the character-level model and return weights + stats."""
    print("Building dataset...")
    contexts, targets = build_dataset(WORDS)
    X = one_hot_encode(contexts)
    N = X.shape[0]
    print(f"  Training samples: {N}")
    print(f"  Input dim:  {INPUT_DIM}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Output dim: {OUTPUT_DIM}")
    print(f"  Total parameters: {INPUT_DIM*HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM*OUTPUT_DIM + OUTPUT_DIM}")
    print()

    # Xavier initialization
    np.random.seed(42)
    W1 = np.random.randn(INPUT_DIM, HIDDEN_DIM) * np.sqrt(2.0 / INPUT_DIM)
    b1 = np.zeros(HIDDEN_DIM)
    W2 = np.random.randn(HIDDEN_DIM, OUTPUT_DIM) * np.sqrt(2.0 / HIDDEN_DIM)
    b2 = np.zeros(OUTPUT_DIM)

    params = [W1, b1, W2, b2]
    optimizer = AdamOptimizer(params, lr=lr)

    print(f"Training for {epochs} epochs (lr={lr}, batch=all)...")
    print("-" * 50)

    for epoch in range(1, epochs + 1):
        # Forward
        probs, cache = forward(X, W1, b1, W2, b2)
        loss = cross_entropy_loss(probs, targets)

        # Backward
        dW1, db1, dW2, db2 = backward(probs, targets, cache, W1, b1, W2, b2)

        # Adam update
        grads = [dW1, db1, dW2, db2]
        W1, b1, W2, b2 = optimizer.step([W1, b1, W2, b2], grads)

        if epoch % 100 == 0 or epoch == 1:
            preds = np.argmax(probs, axis=1)
            acc = (preds == targets).mean()
            print(f"  Epoch {epoch:5d} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

    # Final evaluation
    probs, _ = forward(X, W1, b1, W2, b2)
    final_loss = cross_entropy_loss(probs, targets)
    preds = np.argmax(probs, axis=1)
    final_acc = (preds == targets).mean()
    print("-" * 50)
    print(f"  Final loss:     {final_loss:.4f}")
    print(f"  Final accuracy: {final_acc:.4f}")
    print()

    return W1, b1, W2, b2, final_acc, final_loss


# ---------------------------------------------------------------------------
# Weight export
# ---------------------------------------------------------------------------
def export_weights(W1, b1, W2, b2, accuracy, loss, out_dir="."):
    """Save weights.json and weights_analog.json."""

    weights_path = os.path.join(out_dir, "weights.json")
    analog_path = os.path.join(out_dir, "weights_analog.json")

    # ---- weights.json ----
    data = {
        "architecture": {
            "input": INPUT_DIM,
            "hidden": HIDDEN_DIM,
            "output": OUTPUT_DIM,
            "context_len": CONTEXT_LEN,
            "vocab_size": VOCAB_SIZE,
        },
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
        "vocab": VOCAB,
        "training_accuracy": round(float(accuracy), 4),
        "training_loss": round(float(loss), 4),
    }
    with open(weights_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved weights       -> {weights_path}")

    # ---- weights_analog.json ----
    G_ref = 1e-6    # 1 uS reference conductance
    G_scale = 1e-6  # scale factor
    V_in = 1.0      # input voltage for logic '1'

    def normalize_weights(W):
        """Normalize weight matrix to [-1, 1]."""
        abs_max = np.abs(W).max()
        if abs_max < 1e-12:
            return W
        return W / abs_max

    W1_norm = normalize_weights(W1)
    W2_norm = normalize_weights(W2)

    # Differential conductance pairs
    # G+ = G_ref + w_norm * G_scale
    # G- = G_ref - w_norm * G_scale
    # Net contribution per synapse: V_in * (G+ - G-) = V_in * 2 * w_norm * G_scale
    layer1_Gplus = (G_ref + W1_norm * G_scale).tolist()
    layer1_Gminus = (G_ref - W1_norm * G_scale).tolist()

    layer2_Gplus = (G_ref + W2_norm * G_scale).tolist()
    layer2_Gminus = (G_ref - W2_norm * G_scale).tolist()

    # Biases as millivolt offsets (scale similarly)
    b1_max = np.abs(b1).max() if np.abs(b1).max() > 1e-12 else 1.0
    b2_max = np.abs(b2).max() if np.abs(b2).max() > 1e-12 else 1.0
    layer1_bias_mV = (b1 / b1_max * 100.0).tolist()   # scale to +/-100 mV range
    layer2_bias_mV = (b2 / b2_max * 100.0).tolist()

    analog_data = {
        "G_ref": G_ref,
        "G_scale": G_scale,
        "V_in": V_in,
        "W1_abs_max": float(np.abs(W1).max()),
        "W2_abs_max": float(np.abs(W2).max()),
        "b1_abs_max": float(b1_max),
        "b2_abs_max": float(b2_max),
        "layer1_Gplus": layer1_Gplus,
        "layer1_Gminus": layer1_Gminus,
        "layer1_bias_mV": layer1_bias_mV,
        "layer2_Gplus": layer2_Gplus,
        "layer2_Gminus": layer2_Gminus,
        "layer2_bias_mV": layer2_bias_mV,
    }
    with open(analog_path, "w") as f:
        json.dump(analog_data, f, indent=2)
    print(f"Saved analog weights -> {analog_path}")


# ---------------------------------------------------------------------------
# Prediction / demo
# ---------------------------------------------------------------------------
def load_weights(weights_dir="."):
    """Load weights from weights.json."""
    path = os.path.join(weights_dir, "weights.json")
    with open(path, "r") as f:
        data = json.load(f)
    W1 = np.array(data["W1"])
    b1 = np.array(data["b1"])
    W2 = np.array(data["W2"])
    b2 = np.array(data["b2"])
    return W1, b1, W2, b2


def predict(text, W1=None, b1=None, W2=None, b2=None, weights_dir=None):
    """
    Predict the next character given a string of 3+ characters.

    Uses the last 3 characters as context. If shorter than 3 chars,
    left-pads with spaces.
    """
    if W1 is None:
        wdir = weights_dir if weights_dir else os.path.dirname(os.path.abspath(__file__))
        W1, b1, W2, b2 = load_weights(wdir)

    # Prepare context: take last CONTEXT_LEN chars, pad with spaces if needed
    text = text.lower()
    text = "".join(ch if ch in VOCAB else " " for ch in text)
    if len(text) < CONTEXT_LEN:
        text = " " * (CONTEXT_LEN - len(text)) + text
    ctx = text[-CONTEXT_LEN:]

    # One-hot encode
    X = np.zeros((1, INPUT_DIM), dtype=np.float64)
    for pos in range(CONTEXT_LEN):
        X[0, pos * VOCAB_SIZE + VOCAB[ctx[pos]]] = 1.0

    # Forward pass
    probs, _ = forward(X, W1, b1, W2, b2)
    pred_idx = np.argmax(probs[0])
    pred_char = INV_VOCAB[pred_idx]

    return pred_char


def predict_topk(text, k=5, W1=None, b1=None, W2=None, b2=None, weights_dir=None):
    """Return top-k predicted characters with probabilities."""
    if W1 is None:
        wdir = weights_dir if weights_dir else os.path.dirname(os.path.abspath(__file__))
        W1, b1, W2, b2 = load_weights(wdir)

    text = text.lower()
    text = "".join(ch if ch in VOCAB else " " for ch in text)
    if len(text) < CONTEXT_LEN:
        text = " " * (CONTEXT_LEN - len(text)) + text
    ctx = text[-CONTEXT_LEN:]

    X = np.zeros((1, INPUT_DIM), dtype=np.float64)
    for pos in range(CONTEXT_LEN):
        X[0, pos * VOCAB_SIZE + VOCAB[ctx[pos]]] = 1.0

    probs, _ = forward(X, W1, b1, W2, b2)
    top_indices = np.argsort(probs[0])[::-1][:k]

    results = []
    for idx in top_indices:
        ch = INV_VOCAB[idx]
        display = repr(ch) if ch == " " else ch
        results.append((display, float(probs[0, idx])))
    return results


def demo(weights_dir=None):
    """Run demonstration predictions."""
    wdir = weights_dir if weights_dir else os.path.dirname(os.path.abspath(__file__))
    W1, b1, W2, b2 = load_weights(wdir)

    test_cases = [
        "hel",    # expect 'l' (from "hello")
        "wor",    # expect 'l' (from "world")
        "the",    # could be space or 'r' (from "the", "there", "these", "their")
        "net",    # expect 'w' (from "network")
        "com",    # expect 'p' (from "computer")
        "cir",    # expect 'c' (from "circuit")
        "ana",    # expect 'l' (from "analog")
        "sig",    # expect 'n' (from "signal")
        "des",    # expect 'i' (from "design")
        "tra",    # expect 'n' or 'i' (from "train", "transistor")
        "vol",    # expect 't' (from "voltage")
        "pow",    # expect 'e' (from "power")
        "fre",    # expect 'q' (from "frequency")
        "cur",    # expect 'r' (from "current")
        "  h",    # expect 'e' or 'o' or 'i' (word-initial)
        " ne",    # expect 'u' or 't' or 'v' or 'w' (from "neural", "network", "never", "new")
    ]

    print("=" * 60)
    print("  DEMO: Character Prediction")
    print("=" * 60)
    print(f"  {'Context':<12} {'Predicted':<12} {'Top-5 predictions'}")
    print("-" * 60)

    for ctx in test_cases:
        pred = predict(ctx, W1, b1, W2, b2)
        topk = predict_topk(ctx, k=5, W1=W1, b1=b1, W2=W2, b2=b2)
        topk_str = ", ".join(f"{ch}={p:.2f}" for ch, p in topk)
        display_pred = repr(pred) if pred == " " else pred
        display_ctx = repr(ctx)
        print(f"  {display_ctx:<12} -> {display_pred:<10} [{topk_str}]")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train a tiny character-level neural network for analog circuit implementation."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with existing weights (skip training)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.demo:
        demo(weights_dir=script_dir)
    else:
        W1, b1, W2, b2, accuracy, loss = train(epochs=args.epochs, lr=args.lr)
        export_weights(W1, b1, W2, b2, accuracy, loss, out_dir=script_dir)
        print()
        demo(weights_dir=script_dir)


if __name__ == "__main__":
    main()
