"""
Example/demo script for the perceptron trainer.

This script demonstrates:
- Training the perceptron on a selected logic gate
- Inspecting learned parameters
- Verifying predictions against the truth table

Usage:
    python -m example.demo
or:
    python example/demo.py
"""

import sys
from pathlib import Path
from main import LOGIC_GATES, train, predict

# Ensure the project root is on sys.path so `main` can be imported
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_demo(gate: str = "AND") -> None:
    if gate not in LOGIC_GATES:
        raise ValueError(
            f"Unknown gate: {gate}. Choose from {list(LOGIC_GATES.keys())}"
        )

    print("=" * 60)
    print(f"Perceptron Demo - {gate} gate")
    print("=" * 60)

    # Train perceptron on the chosen gate with default hyperparameters
    w1, w2, bias, converged = train(gate_type=gate, learning_rate=0.1, max_epochs=100)

    print("\nTraining summary:")
    print(f"  Converged: {converged}")
    print(f"  Weights:   w1 = {w1:.4f}, w2 = {w2:.4f}")
    print(f"  Bias:      b  = {bias:.4f}")

    x1_vals, x2_vals, y_vals = LOGIC_GATES[gate]

    print("\nPredictions on truth table:")
    print("  X1  X2 | Expected | Predicted")
    print("  -----|----------|----------")
    for x1, x2, y_true in zip(x1_vals, x2_vals, y_vals):
        y_pred = predict(x1, x2, w1, w2, bias)
        print(f"   {x1}   {x2} |    {y_true}     |     {y_pred}")


if __name__ == "__main__":
    # Default demo on AND gate; adjust as desired.
    run_demo("AND")
