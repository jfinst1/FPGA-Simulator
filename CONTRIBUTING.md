# Contributing to FPGA Simulator

First off, thank you for considering contributing to the FPGA Simulator project! It's people like you that make this project such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible.
* **Provide specific examples to demonstrate the steps**.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include screenshots and animated GIFs** if relevant.
* **Include your environment details** (OS, Python version, CUDA version if applicable).

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**.
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Explain why this enhancement would be useful** to most FPGA Simulator users.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Development Process

### Setting Up Your Development Environment

```bash
# Clone your fork
git clone https://github.com/yourusername/fpga-simulator.git
cd fpga-simulator

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all,dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fpga_simulator tests/

# Run specific test file
pytest tests/test_clb.py

# Run with verbose output
pytest -v
```

### Code Style

We use Black for code formatting and Flake8 for linting:

```bash
# Format code
black fpga_simulator/

# Check linting
flake8 fpga_simulator/

# Run both
make lint  # If Makefile is available
```

### Writing Tests

All new features should include tests. We use pytest for testing:

```python
# tests/test_new_feature.py
import pytest
from fpga_simulator import NewFeature

def test_new_feature_basic():
    """Test basic functionality of new feature."""
    feature = NewFeature()
    result = feature.process(input_data)
    assert result == expected_output

def test_new_feature_edge_case():
    """Test edge cases."""
    feature = NewFeature()
    with pytest.raises(ValueError):
        feature.process(invalid_input)

@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
def test_new_feature_gpu():
    """Test GPU-specific functionality."""
    feature = NewFeature(use_gpu=True)
    # ... GPU-specific tests
```

### Documentation

* Use docstrings for all public modules, functions, classes, and methods.
* Follow Google style for docstrings.
* Update the README.md if you change functionality.
* Add examples to the `examples/` directory for new features.

Example docstring:

```python
def calculate_propagation_delay(distance: float, speed: float = 3e8) -> float:
    """Calculate signal propagation delay.
    
    Args:
        distance: Distance in meters.
        speed: Signal propagation speed in m/s. Defaults to speed of light.
        
    Returns:
        Propagation delay in seconds.
        
    Raises:
        ValueError: If distance is negative or speed is zero.
        
    Example:
        >>> delay = calculate_propagation_delay(0.1)
        >>> print(f"Delay: {delay*1e9:.2f} ns")
        Delay: 0.33 ns
    """
```

## Project Structure Guidelines

When adding new features, follow the existing project structure:

* Core FPGA components go in `fpga_simulator.py`
* Visualization code goes in `fpga_visualizer.py` (if created)
* Web interface code goes in `static/` and `templates/`
* Examples go in `examples/`
* Tests go in `tests/` with matching structure

## Commit Message Guidelines

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line

Example:
```
Add quantum error correction to QPU

- Implement surface code for error correction
- Add noise models for quantum gates
- Update documentation with QEC examples

Fixes #123
```

## Areas We're Looking For Help

* **HDL Support**: Adding VHDL and SystemVerilog parsers
* **Quantum Algorithms**: Implementing more quantum algorithms (Shor's, QAOA, etc.)
* **Visualization**: Improving the web interface with 3D visualization
* **Performance**: Optimizing critical paths and GPU kernels
* **Documentation**: Tutorials, examples, and educational content
* **Testing**: Increasing test coverage and adding integration tests

## Questions?

Feel free to open an issue with the "question" label or reach out to the maintainers directly.

Thank you for contributing! 🎉