#!/usr/bin/env python3
"""
Train a small MLP on digit images for analog crossbar implementation.

Supports two modes:
    --mode 8x8:   sklearn 8x8 digits (1,797 samples, 64 inputs)
    --mode 14x14: downsampled MNIST (10,000 samples, 196 inputs)  [DEFAULT]

Architecture:
    Input:  196 pixels (14x14) or 64 pixels (8x8)
    Layer1: Linear N->hidden, ReLU
    Layer2: Linear hidden->10
    Output: 10 classes (digits 0-9)

Usage:
    python train_model.py                        # 14x14, hidden=64, 2000 epochs
    python train_model.py --mode 8x8 --hidden 32 # 8x8 mode
    python train_model.py --demo                  # Run demo with existing weights
"""

import argparse
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIM = 10  # digits 0-9


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_digits_8x8():
    """Load 8x8 digits from sklearn or bundled file."""
    bundled_path = os.path.join(SCRIPT_DIR, "digits_dataset.json")
    try:
        from sklearn.datasets import load_digits
        digits = load_digits()
        X = digits.data / 16.0
        y = digits.target
        return X, y, 8
    except ImportError:
        pass
    if os.path.exists(bundled_path):
        with open(bundled_path, "r") as f:
            data = json.load(f)
        return np.array(data["X"]), np.array(data["y"]), 8
    raise FileNotFoundError("No 8x8 dataset. Install scikit-learn: pip install scikit-learn")


def load_mnist_14x14():
    """Load MNIST downsampled to 14x14 from bundled file or sklearn."""
    bundled_path = os.path.join(SCRIPT_DIR, "mnist_14x14.npz")
    if os.path.exists(bundled_path):
        data = np.load(bundled_path)
        return data["X"], data["y"], 14

    # Generate from sklearn fetch_openml
    print("  Downloading MNIST (first time only)...")
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X_full = mnist.data.astype(np.float64) / 255.0
        y_full = mnist.target.astype(int)
    except Exception as e:
        raise RuntimeError(f"Could not load MNIST: {e}. Run with --mode 8x8 instead.")

    # Downsample 28x28 -> 14x14 via 2x2 average pooling
    X_28 = X_full.reshape(-1, 28, 28)
    X_14 = X_28.reshape(-1, 14, 2, 14, 2).mean(axis=(2, 4))
    X_flat = X_14.reshape(-1, 196)

    # Subsample to keep it manageable: 8000 train + 2000 test
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X_flat))[:10000]
    X_sub = X_flat[indices]
    y_sub = y_full[indices]

    # Save bundled
    np.savez_compressed(bundled_path, X=X_sub, y=y_sub)
    print(f"  Saved bundled MNIST 14x14 -> {bundled_path} ({len(X_sub)} samples)")

    return X_sub, y_sub, 14


def load_dataset(mode="14x14"):
    """Load digit dataset based on mode."""
    if mode == "8x8":
        return load_digits_8x8()
    else:
        return load_mnist_14x14()


def save_dataset_bundle(X, y, mode="8x8"):
    """Save dataset as bundled file."""
    if mode == "8x8":
        path = os.path.join(SCRIPT_DIR, "digits_dataset.json")
        data = {"X": X.tolist(), "y": y.tolist()}
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"  Saved bundled dataset -> {path}")
    else:
        path = os.path.join(SCRIPT_DIR, "mnist_14x14.npz")
        np.savez_compressed(path, X=X, y=y)
        print(f"  Saved bundled dataset -> {path}")


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split dataset into train/test sets."""
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    indices = rng.permutation(N)
    n_test = int(N * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------
def relu(z):
    return np.maximum(0, z)


def relu_deriv(z):
    return (z > 0).astype(z.dtype)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, targets):
    N = probs.shape[0]
    log_probs = -np.log(probs[np.arange(N), targets] + 1e-12)
    return log_probs.mean()


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------
class AdamOptimizer:
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
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
# Forward / backward
# ---------------------------------------------------------------------------
def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    probs = softmax(z2)
    cache = (X, z1, a1, z2)
    return probs, cache


def backward(probs, targets, cache, W1, b1, W2, b2):
    X, z1, a1, z2 = cache
    N = X.shape[0]

    dz2 = probs.copy()
    dz2[np.arange(N), targets] -= 1.0
    dz2 /= N

    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_deriv(z1)

    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0)

    return dW1, db1, dW2, db2


def predict(X, W1, b1, W2, b2):
    """Forward pass returning class predictions."""
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return np.argmax(z2, axis=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(hidden_dim=64, epochs=2000, lr=0.005, mode="14x14"):
    print("Loading dataset...")
    X_all, y_all, img_size = load_dataset(mode)
    input_dim = X_all.shape[1]
    X_train, y_train, X_test, y_test = train_test_split(X_all, y_all)

    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    print(f"  Mode: {mode} ({img_size}x{img_size})")
    print(f"  Train: {N_train}, Test: {N_test}")
    print(f"  Input: {input_dim}, Hidden: {hidden_dim}, Output: {OUTPUT_DIM}")
    print(f"  Parameters: {input_dim * hidden_dim + hidden_dim + hidden_dim * OUTPUT_DIM + OUTPUT_DIM}")
    print()

    # Xavier init
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, OUTPUT_DIM) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(OUTPUT_DIM)

    optimizer = AdamOptimizer([W1, b1, W2, b2], lr=lr)

    print(f"Training for {epochs} epochs (lr={lr})...")
    print("-" * 60)

    best_test_acc = 0
    for epoch in range(1, epochs + 1):
        probs, cache = forward(X_train, W1, b1, W2, b2)
        loss = cross_entropy_loss(probs, y_train)

        dW1, db1, dW2, db2 = backward(probs, y_train, cache, W1, b1, W2, b2)
        W1, b1, W2, b2 = optimizer.step([W1, b1, W2, b2], [dW1, db1, dW2, db2])

        if epoch % 100 == 0 or epoch == 1:
            train_preds = np.argmax(probs, axis=1)
            train_acc = (train_preds == y_train).mean()
            test_preds = predict(X_test, W1, b1, W2, b2)
            test_acc = (test_preds == y_test).mean()
            best_test_acc = max(best_test_acc, test_acc)
            print(f"  Epoch {epoch:5d} | Loss: {loss:.4f} | "
                  f"Train: {train_acc:.4f} | Test: {test_acc:.4f}")

    # Final eval
    test_preds = predict(X_test, W1, b1, W2, b2)
    test_acc = (test_preds == y_test).mean()
    train_preds = predict(X_train, W1, b1, W2, b2)
    train_acc = (train_preds == y_train).mean()

    print("-" * 60)
    print(f"  Final train accuracy: {train_acc:.4f}")
    print(f"  Final test accuracy:  {test_acc:.4f}")

    # Confusion matrix
    print("\n  Confusion matrix (rows=true, cols=predicted):")
    cm = np.zeros((OUTPUT_DIM, OUTPUT_DIM), dtype=int)
    for t, p in zip(y_test, test_preds):
        cm[t][p] += 1
    print("       " + "".join(f"{i:>5d}" for i in range(10)))
    for i in range(10):
        row = "".join(f"{cm[i][j]:>5d}" for j in range(10))
        print(f"    {i}: {row}")

    return W1, b1, W2, b2, train_acc, test_acc, X_all, y_all, img_size


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def export_weights(W1, b1, W2, b2, train_acc, test_acc, img_size=14, out_dir="."):
    input_dim = W1.shape[0]
    path = os.path.join(out_dir, "weights.json")
    data = {
        "architecture": {
            "input": input_dim,
            "hidden": len(b1),
            "output": OUTPUT_DIM,
            "activation": "relu",
            "task": f"mnist_{img_size}x{img_size}_digit_classification",
            "img_size": img_size,
        },
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
        "train_accuracy": round(float(train_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved weights -> {path}")
    print(f"  W1: {W1.shape}, b1: {b1.shape}")
    print(f"  W2: {W2.shape}, b2: {b2.shape}")
    print(f"  |W1| max: {np.abs(W1).max():.4f}, |W2| max: {np.abs(W2).max():.4f}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def demo(weights_dir=None):
    """Show sample predictions with ASCII art digits."""
    if weights_dir is None:
        weights_dir = SCRIPT_DIR

    path = os.path.join(weights_dir, "weights.json")
    with open(path, "r") as f:
        data = json.load(f)
    W1 = np.array(data["W1"])
    b1 = np.array(data["b1"])
    W2 = np.array(data["W2"])
    b2 = np.array(data["b2"])
    img_size = data.get("architecture", {}).get("img_size", 8)
    mode = "8x8" if img_size == 8 else "14x14"

    X, y, _ = load_dataset(mode)
    _, _, X_test, y_test = train_test_split(X, y)

    print("\n" + "=" * 50)
    print(f"  DEMO: Analog MNIST Digit Classification ({img_size}x{img_size})")
    print("=" * 50)

    # Show 10 random test digits
    rng = np.random.RandomState(123)
    indices = rng.choice(len(X_test), size=10, replace=False)

    correct = 0
    for idx in indices:
        img = X_test[idx]
        true_label = y_test[idx]

        # Forward pass
        z1 = img @ W1 + b1
        a1 = relu(z1)
        logits = a1 @ W2 + b2
        pred = int(np.argmax(logits))

        match = pred == true_label
        correct += match

        # ASCII art
        pixels = img.reshape(img_size, img_size)
        ascii_chars = " .:-=+*#@"
        print(f"\n  True: {true_label}  Predicted: {pred}  {'OK' if match else 'WRONG'}")
        border = "-" * img_size
        print(f"  +{border}+")
        for row in pixels:
            line = ""
            for val in row:
                idx_char = min(int(val * (len(ascii_chars) - 1)), len(ascii_chars) - 1)
                line += ascii_chars[idx_char]
            print(f"  |{line}|")
        print(f"  +{border}+")

        # Top-3 logits
        top3 = np.argsort(logits)[::-1][:3]
        scores = ", ".join(f"{i}({logits[i]:.2f})" for i in top3)
        print(f"  Top-3: {scores}")

    print(f"\n  Demo accuracy: {correct}/{len(indices)} ({100*correct/len(indices):.0f}%)")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train MNIST 8x8 digit classifier for analog circuit.")
    parser.add_argument("--demo", action="store_true", help="Run demo with existing weights")
    parser.add_argument("--mode", choices=["8x8", "14x14"], default="14x14",
                        help="Dataset mode (default: 14x14)")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden layer size (default: 64)")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs (default: 2000)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    parser.add_argument("--save-dataset", action="store_true", help="Save dataset bundle for offline use")
    args = parser.parse_args()

    if args.demo:
        demo(SCRIPT_DIR)
        return

    W1, b1, W2, b2, train_acc, test_acc, X_all, y_all, img_size = train(
        hidden_dim=args.hidden, epochs=args.epochs, lr=args.lr, mode=args.mode
    )
    export_weights(W1, b1, W2, b2, train_acc, test_acc, img_size=img_size, out_dir=SCRIPT_DIR)

    if args.save_dataset:
        save_dataset_bundle(X_all, y_all, mode=args.mode)

    print()
    demo(SCRIPT_DIR)


if __name__ == "__main__":
    main()
