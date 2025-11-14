# Perceptron Repository

## Purpose

This repository implements a minimal, deterministic perceptron trainer - Frank Rosenblatt's foundational artificial neural network from 1958. Built with pure Python and no external dependencies, it serves as an educational tool to demonstrate fundamental machine learning principles through classic logic gate problems.

The perceptron successfully learns linearly separable logic gates (AND, OR, NAND, NOR) with 100% accuracy in a few epochs, while demonstrating the XOR problem limitation that led to the development of multi-layer neural networks.

## General Setup

### Requirements
- **Python**: 3.10 or later (tested on 3.10, 3.11, 3.12)
- **Runtime**: No external dependencies (pure Python)
- **Development**: Optional dependencies for testing and linting

### Installation
```bash
git clone https://github.com/ap0810/perceptron.git
cd perceptron

# For development (optional)
pip install -e ".[dev]"
```

### Basic Usage
```bash
# Train on AND gate with default parameters (zero initialization)
python main.py

# Advanced usage with options
python main.py --gate OR --lr 0.5 --epochs 200 --visualize

# Use industry-standard random initialization
python main.py --gate AND --init random --seed 42
python main.py --gate OR --init random --seed 123 --visualize
```

## Repository Structure

```
perceptron/
├── main.py                    # Core perceptron implementation and CLI
├── README.md                  # Comprehensive documentation
├── pyproject.toml            # Project configuration and dependencies
├── .gitignore                # Git ignore rules
├── .pre-commit-config.yaml   # Pre-commit hooks configuration
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI pipeline
├── .kiro/                    # Kiro steering documentation
│   └── steering/
│       ├── product.md
│       ├── structure.md
│       ├── tech.md
│       └── workflow.md
├── tests/
│   ├── __init__.py
│   └── test_main.py          # Comprehensive test suite (27 unit tests)
└── example/
    └── demo.py               # Example usage demonstration
```

### Key Components

- **`main.py`**: Complete perceptron implementation with:
  - Logic gate definitions (AND, OR, NAND, NOR, XOR)
  - Step activation function
  - Training algorithm with early stopping
  - ASCII decision boundary visualization
  - Comprehensive logging and CLI interface
  - **Weight initialization options**: Zero (pedagogical) vs Random (industry standard)
  - **Reproducibility controls**: Seed support for random initialization

- **`tests/`**: Unit test suite covering:
  - Activation function behavior
  - Prediction accuracy
  - Logic gate validation
  - Training convergence
  - XOR non-convergence (expected)
  - Deterministic reproducibility

- **`example/`**: Demonstration scripts for usage examples

## CI/CD Pipeline

The repository uses GitHub Actions with a comprehensive CI pipeline:

### Workflows (`.github/workflows/ci.yml`)

1. **Lint & Test Job**:
   - **Matrix Testing**: Python 3.10, 3.11, 3.12
   - **Linting**: Ruff for code formatting and style checks
   - **Type Checking**: MyPy for static type analysis
   - **Testing**: pytest with coverage reporting via pytest-cov
   - **Coverage**: Reports test coverage with `--cov=main --cov-report=term-missing`

2. **Security Job**:
   - **Supply Chain Security**: pip-audit for dependency vulnerability scanning
   - **Dependency Review**: GitHub's dependency review action for PRs
   - **Permissions**: Read-only access with security-events write for SARIF uploads

3. **Pre-commit Job**:
   - **Hook Validation**: Runs all pre-commit hooks
   - **Code Quality**: Ensures consistent formatting and standards

4. **Release Job** (on tags):
   - **Package Building**: Creates distribution packages
   - **Artifact Upload**: Stores build artifacts
   - **Triggered**: Only on version tags (`v*`)

### Pre-commit Hooks (`.pre-commit-config.yaml`)

- **Code Quality**:
  - `ruff`: Fast Python linter and formatter
  - `black`: Code formatting
  - `mypy`: Static type checking with strict mode

- **General Checks**:
  - File size limits
  - AST validation
  - YAML syntax checking
  - Trailing whitespace removal
  - End-of-file fixing
  - Merge conflict detection

### Development Tools

- **Linting**: Ruff (configured in `pyproject.toml`)
- **Type Checking**: MyPy with strict mode
- **Testing**: pytest with coverage reporting
- **Formatting**: Black and Ruff format
- **Package Management**: setuptools with pyproject.toml configuration

### Quality Assurance

The repository prioritizes code quality and reproducibility with:
- Comprehensive test coverage (27 unit tests)
- Multi-version Python support (3.10-3.12)
- Strict type checking and linting
- Automated security scanning
- Pre-commit hooks for consistent code quality

### Pedagogical vs Production Design

**Key Design Decision**: The repository offers both pedagogical and production-ready approaches to weight initialization:

- **Default (Zero Initialization)**: Pedagogical choice for deterministic, reproducible results that help learners understand the algorithm without randomness confusion
- **Optional (Random Initialization)**: Industry standard practice with seed control for production-ready ML workflows

**CLI Support**:
```bash
# Pedagogical (default): identical results every run
python main.py --init zero

# Production standard: random initialization with reproducibility
python main.py --init random --seed 42
```

This dual approach helps bridge the gap between educational clarity and real-world ML practices, making the repository valuable for both learning and understanding industry standards.