# Makefile for FPGA Simulator Project

# Python interpreter
PYTHON := python3
PIP := pip3

# Directories
SRC_DIR := .
TEST_DIR := tests
EXAMPLES_DIR := examples
DOCS_DIR := docs

# Files
MAIN_FILE := fpga_simulator.py
WEB_FILE := fpga_web_interface.html

# Default target
.DEFAULT_GOAL := help

# PHONY targets
.PHONY: help install install-all install-dev test coverage lint format clean run-web run-examples docs check-deps

# Help target
help:
	@echo "FPGA Simulator - Available Commands"
	@echo "==================================="
	@echo "  make install       - Install basic dependencies"
	@echo "  make install-all   - Install all dependencies (GPU, Quantum, ML)"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make test          - Run unit tests"
	@echo "  make coverage      - Run tests with coverage report"
	@echo "  make lint          - Run code linting"
	@echo "  make format        - Format code with black"
	@echo "  make clean         - Clean generated files"
	@echo "  make run-web       - Start web interface"
	@echo "  make run-examples  - Run example demonstrations"
	@echo "  make docs          - Generate documentation"
	@echo "  make check-deps    - Check installed dependencies"

# Installation targets
install:
	@echo "Installing basic dependencies..."
	$(PIP) install -r requirements.txt

install-all:
	@echo "Installing all dependencies..."
	$(PIP) install numpy matplotlib flask
	@echo "Installing GPU support (CuPy)..."
	-$(PIP) install cupy-cuda11x
	@echo "Installing Quantum support (Qiskit)..."
	-$(PIP) install qiskit qiskit-aer
	@echo "Installing ML support (scikit-learn)..."
	-$(PIP) install scikit-learn scipy
	@echo "Installing visualization extras..."
	-$(PIP) install seaborn plotly

install-dev: install
	@echo "Installing development dependencies..."
	$(PIP) install pytest pytest-cov black flake8 sphinx sphinx-rtd-theme

# Testing targets
test:
	@echo "Running tests..."
	@if [ -d "$(TEST_DIR)" ]; then \
		$(PYTHON) -m pytest $(TEST_DIR) -v; \
	else \
		echo "No tests directory found. Creating example test..."; \
		mkdir -p $(TEST_DIR); \
		echo "def test_import():\n    import fpga_simulator\n    assert True" > $(TEST_DIR)/test_basic.py; \
		$(PYTHON) -m pytest $(TEST_DIR) -v; \
	fi

coverage:
	@echo "Running tests with coverage..."
	@if [ -d "$(TEST_DIR)" ]; then \
		$(PYTHON) -m pytest --cov=fpga_simulator --cov-report=html --cov-report=term $(TEST_DIR); \
		echo "Coverage report generated in htmlcov/index.html"; \
	else \
		echo "No tests directory found."; \
	fi

# Code quality targets
lint:
	@echo "Running flake8 linter..."
	@flake8 $(SRC_DIR) --exclude=venv,__pycache__ --max-line-length=100 --ignore=E501,W503 || true
	@echo "Checking for common issues..."
	@$(PYTHON) -m py_compile $(MAIN_FILE)

format:
	@echo "Formatting code with black..."
	@black $(SRC_DIR) --line-length=100 --exclude=venv || echo "Black not installed. Run: make install-dev"

# Clean target
clean:
	@echo "Cleaning generated files..."
	@rm -rf __pycache__ */__pycache__ *.pyc */*.pyc
	@rm -rf .pytest_cache .coverage htmlcov
	@rm -rf *.log *.png *.json
	@rm -rf build dist *.egg-info
	@echo "Clean complete."

# Run targets
run-web:
	@echo "Starting FPGA Simulator Web Interface..."
	@echo "======================================="
	@if [ -f "app_simple.py" ]; then \
		$(PYTHON) app_simple.py; \
	elif [ -f "app.py" ]; then \
		$(PYTHON) app.py; \
	else \
		echo "No Flask app found. Please ensure app.py or app_simple.py exists."; \
	fi

run-examples:
	@echo "Running FPGA Simulator Examples..."
	@echo "================================="
	@if [ -f "$(EXAMPLES_DIR)/basic_example.py" ]; then \
		$(PYTHON) $(EXAMPLES_DIR)/basic_example.py; \
	else \
		echo "Running basic demo from main file..."; \
		$(PYTHON) $(MAIN_FILE); \
	fi

run-neural:
	@echo "Running Neural Network Example..."
	@if [ -f "$(EXAMPLES_DIR)/neural_network.py" ]; then \
		$(PYTHON) $(EXAMPLES_DIR)/neural_network.py; \
	else \
		echo "Neural network example not found."; \
	fi

run-quantum:
	@echo "Running Quantum Algorithm Examples..."
	@if [ -f "$(EXAMPLES_DIR)/quantum_search.py" ]; then \
		$(PYTHON) $(EXAMPLES_DIR)/quantum_search.py; \
	else \
		echo "Quantum examples not found."; \
	fi

# Documentation target
docs:
	@echo "Generating documentation..."
	@if command -v sphinx-build &> /dev/null; then \
		mkdir -p $(DOCS_DIR); \
		sphinx-quickstart -q -p "FPGA Simulator" -a "Your Name" -v "1.0" --ext-autodoc --ext-viewcode --makefile $(DOCS_DIR); \
		$(MAKE) -C $(DOCS_DIR) html; \
		echo "Documentation generated in $(DOCS_DIR)/_build/html/index.html"; \
	else \
		echo "Sphinx not installed. Run: make install-dev"; \
	fi

# Dependency check
check-deps:
	@echo "Checking installed dependencies..."
	@echo "================================="
	@$(PYTHON) -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>/dev/null || echo "✗ NumPy: Not installed"
	@$(PYTHON) -c "import matplotlib; print('✓ Matplotlib:', matplotlib.__version__)" 2>/dev/null || echo "✗ Matplotlib: Not installed"
	@$(PYTHON) -c "import flask; print('✓ Flask:', flask.__version__)" 2>/dev/null || echo "✗ Flask: Not installed"
	@$(PYTHON) -c "import cupy; print('✓ CuPy (GPU):', cupy.__version__)" 2>/dev/null || echo "✗ CuPy (GPU): Not installed"
	@$(PYTHON) -c "import qiskit; print('✓ Qiskit:', qiskit.__version__)" 2>/dev/null || echo "✗ Qiskit: Not installed"
	@$(PYTHON) -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)" 2>/dev/null || echo "✗ scikit-learn: Not installed"
	@echo ""
	@$(PYTHON) -c "from fpga_simulator import GPU_AVAILABLE, QUANTUM_AVAILABLE, ML_AVAILABLE; \
		print('Feature Status:'); \
		print('  GPU:', '✓ Available' if GPU_AVAILABLE else '✗ Not available'); \
		print('  Quantum:', '✓ Available' if QUANTUM_AVAILABLE else '✗ Not available'); \
		print('  ML:', '✓ Available' if ML_AVAILABLE else '✗ Not available')"

# Development workflow targets
dev-setup: install-dev
	@echo "Setting up development environment..."
	@git config core.hooksPath .githooks 2>/dev/null || true
	@echo "Development environment ready!"

quick-test:
	@echo "Running quick smoke test..."
	@$(PYTHON) -c "import fpga_simulator; print('✓ Import successful')"
	@$(PYTHON) -c "from fpga_simulator import AdvancedFPGAFabric; \
		fpga = AdvancedFPGAFabric(2, 2); \
		print('✓ FPGA creation successful')"

benchmark:
	@echo "Running performance benchmarks..."
	@$(PYTHON) -c "from fpga_advanced_demo import benchmark_parallel_vs_sequential; \
		from fpga_simulator import AdvancedFPGAFabric; \
		fpga = AdvancedFPGAFabric(8, 8); \
		benchmark_parallel_vs_sequential(fpga, 10)" 2>/dev/null || \
		echo "Benchmark not available. Ensure fpga_advanced_demo.py exists."

# Docker targets (optional)
docker-build:
	@echo "Building Docker image..."
	@if [ -f "Dockerfile" ]; then \
		docker build -t fpga-simulator .; \
	else \
		echo "Creating Dockerfile..."; \
		echo "FROM python:3.9" > Dockerfile; \
		echo "WORKDIR /app" >> Dockerfile; \
		echo "COPY requirements.txt ." >> Dockerfile; \
		echo "RUN pip install -r requirements.txt" >> Dockerfile; \
		echo "COPY . ." >> Dockerfile; \
		echo "CMD [\"python\", \"app_simple.py\"]" >> Dockerfile; \
		docker build -t fpga-simulator .; \
	fi

docker-run:
	@echo "Running FPGA Simulator in Docker..."
	@docker run -p 5000:5000 fpga-simulator

# Release targets
dist:
	@echo "Creating distribution package..."
	@$(PYTHON) setup.py sdist bdist_wheel

upload-test:
	@echo "Uploading to TestPyPI..."
	@twine upload --repository testpypi dist/*

upload:
	@echo "Uploading to PyPI..."
	@twine upload dist/*

# Initialize project structure
init-project:
	@echo "Initializing project structure..."
	@mkdir -p $(TEST_DIR) $(EXAMPLES_DIR) $(DOCS_DIR) static templates
	@touch $(TEST_DIR)/__init__.py
	@echo "Project structure created!"

# Print project info
info:
	@echo "FPGA Simulator Project Information"
	@echo "================================="
	@echo "Main file: $(MAIN_FILE)"
	@echo "Python version: $(shell $(PYTHON) --version)"
	@echo "Current directory: $(shell pwd)"
	@echo "Files: $(shell ls -1 *.py 2>/dev/null | wc -l) Python files"
	@echo "Examples: $(shell ls -1 $(EXAMPLES_DIR)/*.py 2>/dev/null | wc -l) example files"