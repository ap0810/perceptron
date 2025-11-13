## Tech Stack

### Runtime
- Python 3.10+ (tested on 3.10, 3.11, 3.12)
- Pure Python standard library only - no external runtime dependencies

### Development Tools
- **Testing**: pytest >= 7.0, pytest-cov >= 4.0
- **Linting/Formatting**: ruff (configured in pyproject.toml)
- **Type Checking**: mypy (strict mode)
- **Pre-commit**: pre-commit hooks for automated checks
- **Build System**: setuptools >= 61.0

### Configuration
- `pyproject.toml`: Project metadata, dependencies, and tool configuration
- `.pre-commit-config.yaml`: Pre-commit hooks for code quality
- Ruff configured with line-length=88, target Python 3.12, Black-compatible formatting

## Common Commands

### Running the Application
```bash
# Train on AND gate (default)
python main.py

# Train on specific gate with custom parameters
python main.py --gate OR --lr 0.5 --epochs 200

# Show decision boundary visualization
python main.py --gate NAND --visualize

# Quiet mode (errors only)
python main.py --quiet
```

### Testing
```bash
# Run all tests
pytest tests/

# Run with coverage report
pytest tests/ --cov=main --cov-report=term-missing

# Run specific test file
pytest tests/test_main.py
```

### Code Quality
```bash
# Lint and auto-fix with ruff
ruff check main.py tests/
ruff format main.py tests/

# Type checking with mypy
mypy main.py

# Run all pre-commit hooks
pre-commit run --all-files
```

### Installation
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install only test dependencies
pip install pytest pytest-cov
```

### Demo/Example
```bash
# Run example demo script
python -m example.demo
# or
python example/demo.py
```
