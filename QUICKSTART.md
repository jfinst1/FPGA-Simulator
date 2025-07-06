# 🚀 FPGA Simulator - Quick Start Guide

Get up and running with the FPGA Simulator in 5 minutes!

## 📦 Installation (Choose One)

### Option A: Minimal Setup (Recommended for beginners)
```bash
# Just the basics - no GPU/Quantum/ML
pip install numpy matplotlib flask
```

### Option B: Full Setup (All features)
```bash
# Using make (if you have it)
make install-all

# Or manually install everything
pip install numpy matplotlib flask
pip install cupy-cuda11x    # GPU support (optional)
pip install qiskit          # Quantum support (optional)
pip install scikit-learn    # ML support (optional)
```

## 🎮 Three Ways to Start

### 1. Web Interface (Easiest!)
```bash
python app_simple.py
```
Open your browser to `http://localhost:5000`

**What you can do:**
- Click on CLBs to activate them
- Inject faults and watch recovery
- Run quantum algorithms
- Monitor real-time performance

### 2. Interactive Examples (Best for learning)
```bash
python examples/basic_example.py
```

This gives you a menu:
```
1. Basic CLB Operations
2. Block RAM Usage
3. DSP Operations
4. Parallel Processing
5. HDL Compilation
6. Power and Noise
7. Fault Tolerance
8. Run all examples
```

### 3. Python Script (For developers)
```python
# simple_demo.py
from fpga_simulator import AdvancedFPGAFabric
import numpy as np

# Create an FPGA
fpga = AdvancedFPGAFabric(rows=4, cols=4)

# Configure a CLB as an AND gate
clb = fpga.clbs[0][0]
and_gate = [0]*15 + [1]  # Only output 1 when all inputs are 1
clb.configure({'lut': and_gate})

# Test it
inputs = np.array([1, 1, 0, 0])  # First two bits matter
result = clb.evaluate(inputs)
print(f"AND(1,1) = {result.value}")  # Should print 1
```

## 🔥 Cool Things to Try

### 1. Quantum Search (Find a number instantly)
```python
from fpga_simulator import AdvancedFPGAFabric

fpga = AdvancedFPGAFabric(rows=4, cols=4)

# Find the number 42 in a list of 64 numbers
result = fpga.run_quantum_accelerated_search(
    64,  # Search space size
    lambda x: x == 42  # What we're looking for
)
print(f"Found it at position: {result}")
```

### 2. Neural Network on FPGA
```bash
python examples/neural_network.py
```
Watch it learn XOR and recognize patterns!

### 3. Signal Processing with DSP
```python
# Noise removal example
fpga = AdvancedFPGAFabric(rows=4, cols=4, num_dsps=2)
dsp = fpga.dsps[0]

# Create noisy signal
import numpy as np
clean = np.sin(np.linspace(0, 10, 100))
noise = np.random.normal(0, 0.1, 100)
noisy_signal = clean + noise

# Filter it
coeffs = np.ones(5) / 5  # Simple moving average
filtered = dsp.fir_filter(noisy_signal, coeffs)
```

### 4. Compile Real Verilog Code
```python
verilog = """
module blinker(clk, led);
    input clk;
    output led;
    reg [23:0] counter;
    
    always @(posedge clk) begin
        counter <= counter + 1;
    end
    
    assign led = counter[23];
endmodule
"""

bitstream = fpga.compile_hdl(verilog)
print(f"Generated bitstream with {len(bitstream['clbs'])} CLBs")
```

## 🎯 What Each Example Shows

| Example File | What You'll Learn | Cool Features |
|-------------|-------------------|---------------|
| `basic_example.py` | FPGA fundamentals | Interactive menu, all core features |
| `neural_network.py` | AI acceleration | XOR learning, pattern recognition |
| `quantum_search.py` | Quantum algorithms | Grover's search, QFT, VQE |
| `adder.v` | HDL compilation | Real Verilog to bitstream |

## 💡 Pro Tips

1. **Start Simple**: Run `basic_example.py` first
2. **No GPU?** Everything still works, just slower
3. **Web UI**: Best for visualization and playing around
4. **Explore**: Check out the noise simulation - it's realistic!

## 🆘 Quick Troubleshooting

**"Import error"**
```bash
# Make sure you're in the right directory
ls fpga_simulator.py  # Should see this file
```

**"GPU not available"**
```bash
# That's OK! It still works without GPU
# To enable GPU: pip install cupy-cuda11x
```

**"Quantum features not available"**
```bash
# Optional feature. To enable:
pip install qiskit
```

**Web interface won't start**
```bash
# Try the simple server instead
python app_simple.py

# Or just open fpga_web_interface.html in your browser
```

## 📚 Next Steps

1. **Run all examples**: `python examples/basic_example.py` → Choose option 8
2. **Read the code**: Examples are well-commented
3. **Modify examples**: Try changing LUT values, network sizes, etc.
4. **Check the tests**: `pytest tests/` to see how components work
5. **Join the community**: Star on GitHub and contribute!

## 🎉 You're Ready!

You now know enough to:
- Create and configure FPGAs
- Run quantum algorithms
- Accelerate neural networks
- Process signals with DSP blocks
- Compile HDL to bitstream

Have fun exploring! 🚀