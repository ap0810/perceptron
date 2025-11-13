## Product Overview

A minimal, deterministic implementation of the perceptron - Frank Rosenblatt's foundational artificial neural network from 1958. Built with pure Python and no external dependencies.

The project trains a perceptron on classic logic gate problems (AND, OR, NAND, NOR) to demonstrate fundamental machine learning principles. Successfully learns linearly separable gates in a few epochs with 100% accuracy. XOR cannot be learned by a single perceptron due to its non-linear separation requirement.

Key features:
- Pure Python implementation (no NumPy or external runtime dependencies)
- Deterministic training with zero-initialized weights for reproducibility
- Command-line interface with configurable learning rate, epochs, and gate type
- Comprehensive logging to both console and file with timestamps
- Early stopping when perfect accuracy is achieved
- ASCII visualization of learned decision boundaries
- Full test suite with 27 unit tests

This is an educational/demo project prioritizing clarity and determinism over performance.
