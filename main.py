#!/usr/bin/env python3
# =============================================================================
# MIT License
#
# Copyright (c) 2025 Adrian Paredez
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================
"""
Perceptron Trainer
------------------
Author: Adrian Paredez
Version: 0.1.0
License: MIT

A complete, minimal, and deterministic implementation of the *perceptron* |
the first artificial neural network model introduced by **Frank Rosenblatt (1958)**.

This script demonstrates the fundamental principles of machine learning and
binary classification using only Python's standard library | no dependencies.

───────────────────────────────────────────────────────────────────────────────
Conceptual Overview
───────────────────────────────────────────────────────────────────────────────
The perceptron is a *linear classifier* that learns a set of weights (w₁, w₂)
and a bias term (b) to map input data to binary outputs (0 or 1). It learns by
iteratively adjusting its parameters to minimize misclassifications on labeled
training data.

The perceptron learning rule can be expressed as:

    wᵢ ← wᵢ + η * (y - ŷ) * xᵢ
    b  ← b  + η * (y - ŷ)

where:
- wᵢ : weight for input feature xᵢ
- b  : bias term
- η  : learning rate (controls the step size of updates)
- y  : ground-truth label
- ŷ  : model prediction (0 or 1)
- xᵢ : input value

In this example, the perceptron learns logic gates | simple but pedagogically
valuable binary classification tasks. The model converges in a few epochs
and reaches 100% accuracy for linearly separable problems (AND, OR, NAND, NOR).

**Important:** XOR is NOT linearly separable and cannot be learned by a single
perceptron. This fundamental limitation led to the development of multi-layer
neural networks.

───────────────────────────────────────────────────────────────────────────────
Execution Features
───────────────────────────────────────────────────────────────────────────────
- **Pure Python:** No NumPy or external libraries required.
- **Deterministic:** Fixed initialization ensures reproducible results.
- **Command-Line Interface:** Adjustable learning rate, epochs, and gate type.
- **Comprehensive Logging:** Logs both to console and file with timestamps.
- **Early Stopping:** Automatically halts once perfect accuracy is achieved.
- **Validation:** Separate test set evaluation to demonstrate generalization.
- **Visualization:** ASCII decision boundary plot for intuitive understanding.
- **Input Validation:** Robust error handling for CLI parameters.
───────────────────────────────────────────────────────────────────────────────
"""

__author__ = "Adrian Paredez"
__version__ = "0.1.0"

import logging
import argparse
import sys
import random
from typing import Tuple, List, Optional


# -----------------------------------------------------------------------------
# Logic Gate Definitions
# -----------------------------------------------------------------------------
LOGIC_GATES = {
    "AND": ([0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]),
    "OR": ([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1]),
    "NAND": ([0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 1, 0]),
    "NOR": ([0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 0]),
    "XOR": ([0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]),  # Not linearly separable!
}


# -----------------------------------------------------------------------------
# Weight Initialization
# -----------------------------------------------------------------------------
def initialize_weights(
    init_type: str = "zero", seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Initialize perceptron weights using different strategies.

    Args:
        init_type (str): Initialization method ("zero" or "random")
        seed (int): Random seed for reproducible random initialization

    Returns:
        tuple: (w1, w2, bias) - Initial weight values

    Raises:
        ValueError: If init_type is not "zero" or "random"
    """
    if init_type == "zero":
        # Pedagogical choice: deterministic, identical results every run
        return 0.0, 0.0, 0.0
    elif init_type == "random":
        # Industry standard: random initialization with optional seed
        if seed is not None:
            random.seed(seed)
        w1 = random.uniform(-0.5, 0.5)
        w2 = random.uniform(-0.5, 0.5)
        bias = random.uniform(-0.5, 0.5)
        return w1, w2, bias
    else:
        raise ValueError(f"Unknown init_type: {init_type}. Use 'zero' or 'random'.")


# -----------------------------------------------------------------------------
# Activation Function
# -----------------------------------------------------------------------------
def step(x: float) -> int:
    """
    Binary step activation function.

    This function defines the perceptron's decision boundary.
    If the weighted input sum (x) is greater than or equal to zero,
    the neuron outputs 1; otherwise, it outputs 0.

    Mathematically:
        step(x) = { 1 if x >= 0
                    0 if x <  0 }

    Args:
        x (float): The weighted sum of inputs plus bias.

    Returns:
        int: 1 if activation threshold is met, else 0.
    """
    return 1 if x >= 0 else 0


# -----------------------------------------------------------------------------
# Prediction Function
# -----------------------------------------------------------------------------
def predict(x1: float, x2: float, w1: float, w2: float, bias: float) -> int:
    """
    Make a prediction using the trained perceptron.

    Args:
        x1, x2: Input features
        w1, w2: Learned weights
        bias: Learned bias term

    Returns:
        int: Binary prediction (0 or 1)
    """
    z = x1 * w1 + x2 * w2 + bias
    return step(z)


# -----------------------------------------------------------------------------
# Visualization Function
# -----------------------------------------------------------------------------
def visualize_decision_boundary(
    w1: float, w2: float, bias: float, X1: List[int], X2: List[int], Y: List[int]
) -> None:
    """
    Display an ASCII visualization of the learned decision boundary.

    The perceptron learns a linear decision boundary defined by:
        w1*x1 + w2*x2 + b = 0

    This function creates a simple 2D grid showing how the perceptron
    classifies different regions of the input space.

    Args:
        w1, w2: Learned weights
        bias: Learned bias term
        X1, X2: Input features from training data
        Y: True labels
    """
    logging.info("")
    logging.info("=" * 60)
    logging.info("DECISION BOUNDARY VISUALIZATION")
    logging.info("=" * 60)
    logging.info("Grid shows perceptron predictions across input space")
    logging.info("'0' = predicts 0 | '1' = predicts 1 | '*' = training point")
    logging.info("")

    # Create a grid from -0.5 to 1.5 for better visualization
    grid_size = 21
    for i in range(grid_size):
        row = ""
        x2_val = 1.5 - (i * 2.0 / (grid_size - 1))  # Top to bottom: 1.5 to -0.5

        for j in range(grid_size):
            x1_val = -0.5 + (j * 2.0 / (grid_size - 1))  # Left to right: -0.5 to 1.5

            # Check if this point is a training sample
            is_training_point = False
            for k in range(len(X1)):
                if abs(X1[k] - x1_val) < 0.05 and abs(X2[k] - x2_val) < 0.05:
                    is_training_point = True
                    break

            if is_training_point:
                row += "*"
            else:
                pred = predict(x1_val, x2_val, w1, w2, bias)
                row += str(pred)

        # Add axis label
        if i == 0:
            row += "  ← x2 = 1.5"
        elif i == grid_size // 2:
            row += "  ← x2 = 0.5"
        elif i == grid_size - 1:
            row += "  ← x2 = -0.5"

        logging.info(row)

    logging.info(" " * 9 + "↑")
    logging.info(" " * 7 + "x1 axis")
    logging.info("")


# -----------------------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------------------
def train(
    gate_type: str = "AND",
    learning_rate: float = 0.1,
    max_epochs: int = 100,
    init_type: str = "zero",
    seed: Optional[int] = None,
) -> Tuple[float, float, float, bool]:
    """
    Train a single-layer perceptron on a specified logic gate dataset.

    The perceptron iteratively updates its weights and bias to correctly
    classify all four possible binary input combinations for the logic function.

    Args:
        gate_type (str): Type of logic gate (AND, OR, NAND, NOR, XOR)
        learning_rate (float): Determines the magnitude of parameter updates.
        max_epochs (int): Maximum number of training passes through the dataset.
        init_type (str): Weight initialization method ("zero" or "random")
        seed (int): Random seed for reproducible random initialization

    Returns:
        tuple: (w1, w2, bias, converged) - Final parameters and convergence status
    """

    if gate_type not in LOGIC_GATES:
        raise ValueError(f"Unknown gate type: {gate_type}")

    # Get dataset for specified logic gate
    X1, X2, Y = LOGIC_GATES[gate_type]

    logging.info("")
    logging.info("=" * 60)
    logging.info(f"TRAINING PERCEPTRON ON {gate_type} LOGIC GATE")
    logging.info("=" * 60)
    logging.info("Truth table:")
    logging.info("  X1  X2 | Y")
    logging.info("  --------|---")
    for i in range(4):
        logging.info(f"   {X1[i]}   {X2[i]}  | {Y[i]}")
    logging.info("")

    # Warn about XOR impossibility
    if gate_type == "XOR":
        logging.warning("⚠️  XOR is NOT linearly separable!")
        logging.warning(
            "⚠️  A single perceptron cannot learn XOR. Training will not converge."
        )
        logging.warning("")

    # Initialize weights using specified method
    w1, w2, bias = initialize_weights(init_type, seed)
    converged = False

    # Log initialization method
    if init_type == "zero":
        logging.info("Initialization: Zero weights (pedagogical/deterministic)")
    else:
        seed_info = f" with seed {seed}" if seed is not None else " (no seed)"
        logging.info(f"Initialization: Random weights{seed_info} (industry standard)")
    logging.info(f"Initial weights: w1={w1:.3f}, w2={w2:.3f}, bias={bias:.3f}")
    logging.info("")

    # ────────────────────────────────────────────────────────────────────────
    # Training Loop
    # ────────────────────────────────────────────────────────────────────────
    for epoch in range(max_epochs):
        total_error = 0
        correct_predictions = 0

        # Iterate through each sample in the training set.
        for i in range(4):
            # Compute the linear combination of inputs and bias.
            z = X1[i] * w1 + X2[i] * w2 + bias

            # Apply the activation function to obtain binary output.
            y_hat = step(z)

            # Compute the prediction error.
            error = Y[i] - y_hat

            # Update parameters using the perceptron learning rule.
            w1 += learning_rate * error * X1[i]
            w2 += learning_rate * error * X2[i]
            bias += learning_rate * error

            # Accumulate absolute error for reporting.
            total_error += abs(error)

        # ────────────────────────────────────────────────────────────────────
        # Epoch Evaluation
        # ────────────────────────────────────────────────────────────────────
        for i in range(4):
            prediction = predict(X1[i], X2[i], w1, w2, bias)
            if prediction == Y[i]:
                correct_predictions += 1

        # Compute accuracy as a percentage of correct classifications.
        accuracy = (correct_predictions / len(Y)) * 100

        # Log epoch performance.
        logging.info(
            f"Epoch {epoch + 1:3d} | Error: {total_error} | "
            f"Accuracy: {accuracy:5.1f}% | "
            f"w1={w1:6.3f}, w2={w2:6.3f}, b={bias:6.3f}"
        )

        # Implement early stopping once 100% accuracy is achieved.
        if accuracy == 100:
            logging.info("")
            logging.info(f"✓ Converged in {epoch + 1} epochs!")
            converged = True
            break

    if not converged:
        logging.info("")
        logging.info(f"✗ Did not converge after {max_epochs} epochs.")
        if gate_type == "XOR":
            logging.info("  (This is expected for XOR - not linearly separable)")

    # ────────────────────────────────────────────────────────────────────────
    # Final Model Summary
    # ────────────────────────────────────────────────────────────────────────
    logging.info("")
    logging.info("=" * 60)
    logging.info("FINAL MODEL PARAMETERS")
    logging.info("=" * 60)
    logging.info(f"w1 = {w1:.6f}")
    logging.info(f"w2 = {w2:.6f}")
    logging.info(f"b  = {bias:.6f}")
    logging.info("")
    logging.info("Decision boundary equation:")
    logging.info(f"  {w1:.3f}*x1 + {w2:.3f}*x2 + {bias:.3f} = 0")
    logging.info("")

    # Show model predictions for each input pattern.
    logging.info("PREDICTIONS ON TRAINING DATA:")
    logging.info("  X1  X2 | Predicted | Actual | Match")
    logging.info("  -------|-----------|--------|------")
    for i in range(4):
        y_hat = predict(X1[i], X2[i], w1, w2, bias)
        match = "✓" if y_hat == Y[i] else "✗"
        logging.info(f"   {X1[i]}   {X2[i]}  |     {y_hat}     |   {Y[i]}    | {match}")

    # Return model parameters for reuse or analysis.
    return w1, w2, bias, converged


# -----------------------------------------------------------------------------
# Test Function
# -----------------------------------------------------------------------------
def test(
    w1: float,
    w2: float,
    bias: float,
    test_data: Tuple[List[int], List[int], List[int]],
    description: str = "test set",
) -> None:
    """
    Evaluate the trained perceptron on a test dataset.

    Args:
        w1, w2: Learned weights
        bias: Learned bias term
        test_data: Tuple of (X1, X2, Y) test samples
        description: Name/description of the test set
    """
    X1, X2, Y = test_data
    correct = 0

    logging.info("")
    logging.info("=" * 60)
    logging.info(f"EVALUATION ON {description.upper()}")
    logging.info("=" * 60)

    for i in range(len(X1)):
        y_hat = predict(X1[i], X2[i], w1, w2, bias)
        match = "✓" if y_hat == Y[i] else "✗"
        logging.info(
            f"Input: ({X1[i]}, {X2[i]}) → Predicted: {y_hat} | Actual: {Y[i]} {match}"
        )
        if y_hat == Y[i]:
            correct += 1

    accuracy = (correct / len(Y)) * 100
    logging.info("")
    logging.info(f"Test Accuracy: {accuracy:.1f}% ({correct}/{len(Y)} correct)")


# -----------------------------------------------------------------------------
# Command-Line Interface (CLI)
# -----------------------------------------------------------------------------
def main() -> int:
    """
    Entry point for command-line execution.

    Enables users to configure the learning rate, number of epochs, and logic
    gate type directly via command-line arguments. The script logs results to
    both stdout and a timestamped file ("training.log") for reproducibility.
    """
    parser = argparse.ArgumentParser(
        description="Train a minimal perceptron on logic gate datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --gate AND --lr 0.1 --epochs 10
  %(prog)s --gate OR --lr 0.2 --epochs 20 --visualize
  %(prog)s --gate XOR --lr 0.1 --epochs 100
  %(prog)s --gate AND --init random --seed 42
  %(prog)s --gate OR --init random --seed 123 --visualize

Available gates: AND, OR, NAND, NOR, XOR
Note: XOR is not linearly separable and will not converge.

Initialization methods:
  zero   - Deterministic (same results every run)
  random - Industry standard (use --seed for reproducibility)
        """,
    )
    parser.add_argument(
        "--gate",
        type=str,
        default="AND",
        choices=LOGIC_GATES.keys(),
        help="Logic gate to learn (default: AND)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate η (default: 0.1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Maximum training epochs (default: 100)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show ASCII decision boundary visualization",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Minimal output (errors only)"
    )
    parser.add_argument(
        "--init",
        type=str,
        default="zero",
        choices=["zero", "random"],
        help="Weight initialization method (default: zero for reproducibility)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible random initialization (only with --init random)",
    )

    args = parser.parse_args()

    # ────────────────────────────────────────────────────────────────────────
    # Input Validation
    # ────────────────────────────────────────────────────────────────────────
    if args.lr <= 0:
        print(f"Error: Learning rate must be positive (got {args.lr})", file=sys.stderr)
        sys.exit(1)

    if args.lr > 1:
        print(
            f"Warning: Learning rate {args.lr} is unusually high. "
            "Typical range is 0.001 to 0.5",
            file=sys.stderr,
        )

    if args.epochs < 1:
        print(f"Error: Epochs must be at least 1 (got {args.epochs})", file=sys.stderr)
        sys.exit(1)

    if args.seed is not None and args.init == "zero":
        print("Warning: --seed has no effect with --init zero", file=sys.stderr)

    if args.epochs > 10000:
        print(
            f"Warning: {args.epochs} epochs is very high. Training may take a while.",
            file=sys.stderr,
        )

    # ────────────────────────────────────────────────────────────────────────
    # Logging Configuration
    # ────────────────────────────────────────────────────────────────────────
    log_level = logging.ERROR if args.quiet else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[
            logging.FileHandler("training.log", mode="w"),
            logging.StreamHandler(),
        ],
    )

    # ────────────────────────────────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────────────────────────────────
    try:
        w1, w2, bias, converged = train(
            args.gate, args.lr, args.epochs, args.init, args.seed
        )

        # Visualize decision boundary if requested
        if args.visualize:
            X1, X2, Y = LOGIC_GATES[args.gate]
            visualize_decision_boundary(w1, w2, bias, X1, X2, Y)

        # Test on the same data (for demonstration purposes)
        # In real ML, you'd use separate train/test splits
        test(w1, w2, bias, LOGIC_GATES[args.gate], f"{args.gate} training data")

        # ────────────────────────────────────────────────────────────────────
        # Final Summary
        # ────────────────────────────────────────────────────────────────────
        logging.info("")
        logging.info("=" * 60)
        logging.info("TRAINING COMPLETE")
        logging.info("=" * 60)
        status = "✓ CONVERGED" if converged else "✗ NOT CONVERGED"
        logging.info(f"Status: {status}")
        logging.info(f"Final Parameters: w1={w1:.6f}, w2={w2:.6f}, bias={bias:.6f}")
        logging.info("Results saved to: training.log")
        logging.info("")

        return 0 if converged else 1

    except Exception as e:
        logging.error(f"Error during training: {e}")
        return 1


# -----------------------------------------------------------------------------
# Script Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
