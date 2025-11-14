# Perceptron Trainer

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/ap0810/perceptron/actions/workflows/ci.yml/badge.svg)](https://github.com/ap0810/perceptron/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-via%20pytest--cov-blue)](https://github.com/ap0810/perceptron/actions)

A minimal, deterministic implementation of the [perceptron](https://en.wikipedia.org/wiki/Perceptron) - [Frank Rosenblatt](https://en.wikipedia.org/wiki/Frank_Rosenblatt)'s foundational artificial neural network from 1958. Built with pure Python and no external dependencies.

---

## Contents

- [Overview](#overview)
- [Logic Gates](#logic-gates)
- [Requirements](#requirements)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Testing](#testing)
- [Development](#development)
- [Output](#output)
- [Note](#note)
- [License & Author](#license--author)
- [References](#references)

---

## Overview

> This project trains a perceptron on classic logic gate problems to demonstrate fundamental machine learning principles. The implementation includes comprehensive logging, early stopping, ASCII visualization of decision boundaries, and full validation metrics.

---

## Logic Gates

The perceptron successfully learns linearly separable [logic gates](https://en.wikipedia.org/wiki/Logic_gate) (AND, OR, NAND, NOR) in a few epochs with 100% accuracy. [XOR](https://en.wikipedia.org/wiki/XOR_gate) cannot be learned by a single perceptron due to its non-linear separation requirement.

---

## Requirements

Python 3.10 or later (tested on 3.10, 3.11, 3.12)

**Runtime**: No external dependencies (pure Python)

**Development** (optional):

- `pytest>=7.0` - Unit testing framework
- `pytest-cov>=4.0` - Coverage reporting

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ap0810/perceptron.git
cd perceptron
```

**For development** (running tests and linting):

```bash
# Using pip with dev dependencies
pip install -e ".[dev]"

# Or install test dependencies directly
pip install pytest pytest-cov
```

---

## Getting Started

To train a perceptron on the AND logic gate with default parameters:

```bash
python main.py
```

The trainer will:

1. Initialize weights to zero for deterministic results (see [Why Zero Initialization?](#why-zero-initialization))
2. Train on the AND gate dataset
3. Display real-time training progress to console
4. Log results to `training.log`
5. Visualize the learned decision boundary

Results will show epoch-by-epoch accuracy and final model parameters.

---

## Usage

Command-line interface for advanced configuration:

```bash
python main.py [--gate GATE] [--lr LR] [--epochs EPOCHS] [--visualize] [--quiet] [--init INIT] [--seed SEED]
```

Arguments:

- `--gate GATE` : Logic gate (AND, OR, NAND, NOR, XOR); default: AND
- `--lr LR` : Learning rate η; default: 0.1
- `--epochs EPOCHS` : Maximum epochs; default: 100
- `--visualize` : Show ASCII decision boundary visualization
- `--quiet` : Minimal output (errors only)
- `--init INIT` : Weight initialization method (zero, random); default: zero
- `--seed SEED` : Random seed for reproducible random initialization

Examples:

```bash
python main.py --gate OR --lr 0.5
python main.py --gate NAND --epochs 200
python main.py --gate AND --init random --seed 42
python main.py --gate OR --init random --seed 123 --visualize
```

---

## Testing

The project includes a comprehensive test suite with 27 unit tests covering:

- Activation function behavior (step function)
- Prediction accuracy
- Logic gate definitions and data validation
- Training convergence on linearly separable problems
- XOR non-convergence (expected behavior)
- Deterministic reproducibility
- Post-training accuracy validation

Run tests with pytest:

```bash
pytest tests/
```

Run with coverage report:

```bash
pytest tests/ --cov=main --cov-report=term-missing
```

Use `pytest` and `pytest-cov` locally to measure coverage for your environment.

---

## Development

This project prioritizes code quality and reproducibility:

**Linting & Formatting**:

```bash
# Using ruff (configured in pyproject.toml)
ruff check main.py tests/
ruff format main.py tests/
```

**Type Checking** (optional):

```bash
# Using mypy for static type analysis
mypy main.py
```

**Pre-commit Hooks** (recommended):

```bash
# Install pre-commit framework
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**Python Versions**: Tested on Python 3.10, 3.11, 3.12. See `pyproject.toml` for configuration.

---

## Output

The trainer logs progress to console and file with timestamps, displays training/test accuracy metrics, and generates an ASCII visualization of the learned decision boundary.

---

## Why Zero Initialization?

This implementation uses **zero initialization** for pedagogical reasons, which differs from standard ML practices. Here's why and how to switch:

### Pedagogical Choice (Current Default)

**Zero initialization** makes results completely deterministic and reproducible across runs:

```python
# Current approach in main.py
w1, w2, bias = 0.0, 0.0, 0.0  # Always the same starting point
```

**Benefits for learning:**
- Identical results every time (no randomness to confuse beginners)
- Easy to debug and verify against expected outputs
- Clear demonstration of the learning algorithm's behavior
- Simplified testing and validation

### Standard ML Practice (Recommended for Production)

**Random initialization** with controlled seeds is the industry standard:

```python
import random

# Standard approach with reproducible randomness
random.seed(42)  # For reproducibility
w1 = random.uniform(-0.5, 0.5)
w2 = random.uniform(-0.5, 0.5)
bias = random.uniform(-0.5, 0.5)
```

**Benefits for real applications:**
- Breaks symmetry in neural networks
- Improves model diversity and generalization
- Prevents identical neurons in multi-layer networks
- Industry standard practice

### Switching to Random Initialization

To use random initialization while maintaining reproducibility:

```bash
# Use random initialization with reproducible seed
python main.py --init random --seed 42
python main.py --init zero    # Current default (pedagogical)

# Random initialization without seed (different results each run)
python main.py --init random

# Compare the two approaches on the same problem
python main.py --gate AND --init zero --visualize
python main.py --gate AND --init random --seed 42 --visualize
```

**Implementation (now available in `main.py`):**

The `initialize_weights()` function supports both initialization methods:

```python
def initialize_weights(init_type="zero", seed=None):
    """Initialize perceptron weights."""
    if init_type == "zero":
        return 0.0, 0.0, 0.0
    elif init_type == "random":
        if seed is not None:
            random.seed(seed)
        w1 = random.uniform(-0.5, 0.5)
        w2 = random.uniform(-0.5, 0.5)
        bias = random.uniform(-0.5, 0.5)
        return w1, w2, bias
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
```

### Trade-offs Summary

| Aspect | Zero Init (Current) | Random Init (Standard) |
|--------|-------------------|----------------------|
| **Reproducibility** | Perfect (no seed needed) | Good (with seed) |
| **Learning Value** | High (predictable) | Medium (realistic) |
| **Industry Practice** | No | Yes |
| **Debugging** | Easy | Moderate |
| **Scalability** | Poor (symmetry issues) | Good |

### Recommendation

- **For learning**: Use zero initialization (current default)
- **For production**: Switch to random initialization with seeds
- **For research**: Use random initialization with multiple seeds for statistical significance

## Note

> [!NOTE]
> This implementation prioritizes **clarity and determinism** over performance for educational purposes. The zero initialization choice is intentional for teaching, but learners should understand that production ML systems typically use random initialization to break symmetry and improve model diversity.

For more information about the perceptron and its limitations, see the [XOR Problem](https://en.wikipedia.org/wiki/XOR_problem) and [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron).

---

## License & Author

[MIT License](https://opensource.org/licenses/MIT) - Adrian Paredez

GitHub: [ap0810/perceptron](https://github.com/ap0810/perceptron)

---

## References

- [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://en.wikipedia.org/wiki/Perceptron#History) - Frank Rosenblatt (1958)
- [Perceptron - Wikipedia](https://en.wikipedia.org/wiki/Perceptron)
- [XOR Problem - Wikipedia](https://en.wikipedia.org/wiki/XOR_problem)
- [Multilayer Perceptron - Wikipedia](https://en.wikipedia.org/wiki/Multilayer_perceptron)
- [Artificial Neuron - Wikipedia](https://en.wikipedia.org/wiki/Artificial_neuron)

---

**Footnote:** The perceptron was the subject of intense debate in the 1960s following Marvin Minsky and Seymour Papert's book "Perceptrons" (1969), which mathematically proved its limitations on non-linearly separable problems. This critique led to the "AI Winter" but also sparked the development of backpropagation and modern deep learning techniques that eventually superseded it.
