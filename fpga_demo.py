import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

# Import our FPGA simulator (assuming it's in fpga_simulator.py)
# from fpga_simulator import ParallelFPGAFabric, ConfigurableLogicBlock, Signal

def create_adder_bitstream(bits: int = 4) -> dict:
    """Create bitstream for N-bit ripple carry adder"""
    bitstream = {'clbs': [], 'routes': []}
    
    # Full adder truth table
    # Inputs: A, B, Cin | Outputs: Sum, Cout
    full_adder_sum = [0, 1, 1, 0, 1, 0, 0, 1]  # A XOR B XOR Cin
    full_adder_carry = [0, 0, 0, 1, 0, 1, 1, 1]  # AB + BCin + ACin
    
    for i in range(bits):
        # Sum CLB
        bitstream['clbs'].append({
            'position': (i, 0),
            'lut': full_adder_sum * 2  # Extend to 4-input LUT
        })
        
        # Carry CLB
        bitstream['clbs'].append({
            'position': (i, 1),
            'lut': full_adder_carry * 2
        })
        
        # Route carry to next stage
        if i < bits - 1:
            src = i * 2 + 1  # Carry output
            dst = (i + 1) * 2  # Next stage carry input
            bitstream['routes'].append({
                'source': src,
                'destination': dst
            })
    
    return bitstream


def create_neural_network_accelerator(input_size: int = 4, hidden_size: int = 8) -> dict:
    """Create FPGA configuration for neural network inference"""
    bitstream = {'clbs': [], 'routes': []}
    
    # Implement activation function (ReLU) as LUT
    # For 4-bit inputs, approximate ReLU
    relu_lut = []
    for i in range(16):
        # Simple threshold at 8
        relu_lut.append(1 if i >= 8 else 0)
    
    # Configure CLBs for neural network layers
    for i in range(hidden_size):
        # Each neuron as a CLB
        bitstream['clbs'].append({
            'position': (i // 4, i % 4),
            'lut': relu_lut,
            'quantum_gate': 'hadamard' if i % 2 == 0 else None  # Add quantum enhancement
        })
    
    # Create fully connected routing
    for i in range(input_size):
        for j in range(hidden_size):
            bitstream['routes'].append({
                'source': i,
                'destination': input_size + j
            })
    
    return bitstream


def benchmark_parallel_vs_sequential(fpga: 'ParallelFPGAFabric', num_tests: int = 100):
    """Benchmark parallel vs sequential evaluation"""
    print("\n=== Performance Benchmarking ===")
    
    # Generate test data
    test_inputs = {}
    for i in range(fpga.rows):
        for j in range(fpga.cols):
            test_inputs[(i, j)] = np.random.randint(0, 2, 4)
    
    # Sequential evaluation
    start_time = time.time()
    for _ in range(num_tests):
        seq_results = {}
        for pos, inputs in test_inputs.items():
            clb = fpga.clbs[pos[0]][pos[1]]
            seq_results[pos] = clb.evaluate(inputs)
    seq_time = time.time() - start_time
    
    # Parallel evaluation
    start_time = time.time()
    for _ in range(num_tests):
        par_results = fpga.parallel_evaluate(test_inputs)
    par_time = time.time() - start_time
    
    print(f"Sequential Time: {seq_time:.3f}s")
    print(f"Parallel Time: {par_time:.3f}s")
    print(f"Speedup: {seq_time/par_time:.2f}x")
    
    # GPU evaluation if available
    if fpga.use_gpu and GPU_AVAILABLE:
        batch_inputs = np.random.randint(0, 2, (fpga.rows * fpga.cols, 4))
        
        start_time = time.time()
        for _ in range(num_tests):
            gpu_results = fpga.gpu_batch_evaluate(batch_inputs)
        gpu_time = time.time() - start_time
        
        print(f"GPU Batch Time: {gpu_time:.3f}s")
        print(f"GPU Speedup: {seq_time/gpu_time:.2f}x")


def test_quantum_enhanced_logic(fpga: 'ParallelFPGAFabric'):
    """Test quantum-enhanced logic blocks"""
    print("\n=== Quantum Logic Testing ===")
    
    if not QUANTUM_AVAILABLE:
        print("Quantum features not available. Install qiskit to enable.")
        return
    
    # Configure a CLB with quantum enhancement
    quantum_config = {
        'lut': [0, 1, 1, 0, 1, 0, 0, 1] * 2,  # XOR-like behavior
        'quantum_gate': 'hadamard'
    }
    
    fpga.clbs[0][0].configure(quantum_config)
    
    # Test with different inputs
    test_cases = [
        np.array([0, 0, 0, 0]),
        np.array([1, 0, 1, 0]),
        np.array([1, 1, 1, 1])
    ]
    
    for inputs in test_cases:
        signal = fpga.clbs[0][0].evaluate(inputs)
        print(f"Input: {inputs} -> Output: {signal.value}")
        if signal.quantum_state is not None:
            print(f"  Quantum state: {signal.quantum_state[:4]}...")  # Show first 4 amplitudes


def visualize_fpga_utilization(fpga: 'ParallelFPGAFabric'):
    """Visualize FPGA resource utilization"""
    print("\n=== FPGA Utilization Visualization ===")
    
    # Create utilization matrix
    utilization = np.zeros((fpga.rows, fpga.cols))
    
    for i in range(fpga.rows):
        for j in range(fpga.cols):
            clb = fpga.clbs[i][j]
            utilization[i, j] = clb.evaluation_count
    
    # Normalize
    if utilization.max() > 0:
        utilization = utilization / utilization.max()
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(utilization, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Normalized Utilization')
    plt.title('FPGA CLB Utilization Heatmap')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Add grid
    for i in range(fpga.rows + 1):
        plt.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(fpga.cols + 1):
        plt.axvline(j - 0.5, color='gray', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('fpga_utilization.png')
    print("Utilization heatmap saved as 'fpga_utilization.png'")


def test_ml_routing_optimization(fpga: 'ParallelFPGAFabric'):
    """Test ML-based routing optimization"""
    print("\n=== ML Routing Optimization ===")
    
    if not ML_AVAILABLE:
        print("ML features not available. Install scikit-learn to enable.")
        return
    
    # Generate synthetic routing performance data
    for _ in range(200):
        # Simulate routing with random features
        route_features = np.random.rand(10)  # 10 routing features
        delay = np.random.exponential(1.0) * 1e-9  # Random delay
        
        fpga.performance_log.append({
            'route_features': route_features,
            'delay': delay
        })
    
    # Train ML model
    print("Training ML routing optimizer...")
    fpga.optimize_routing_ml()
    
    # Test predictions
    test_features = np.random.rand(10)
    if fpga.ml_router.model is not None:
        predicted_delay = fpga.ml_router.predict_delay(test_features)
        print(f"Predicted routing delay: {predicted_delay*1e9:.2f} ns")


def simulate_digital_filter(fpga: 'ParallelFPGAFabric'):
    """Implement a simple FIR filter on FPGA"""
    print("\n=== Digital Filter Implementation ===")
    
    # Configure CLBs as multiply-accumulate units
    # Simple 4-tap FIR filter
    filter_coeffs = [1, 2, 1, 1]  # Simplified to binary
    
    # Configure CLBs for filter taps
    for i, coeff in enumerate(filter_coeffs):
        # Simple multiplication by shift (power of 2)
        if coeff == 1:
            lut = list(range(16))  # Identity
        elif coeff == 2:
            lut = [(i << 1) & 0xF for i in range(16)]  # Left shift
        else:
            lut = [0] * 16
        
        fpga.clbs[0][i].configure({'lut': lut})
    
    # Process signal samples
    input_signal = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    output_signal = []
    
    for i in range(len(input_signal) - len(filter_coeffs) + 1):
        # Get filter window
        window = input_signal[i:i+len(filter_coeffs)]
        
        # Process through filter
        acc = 0
        for j, sample in enumerate(window):
            inputs = np.array([sample, 0, 0, 0])
            result = fpga.clbs[0][j].evaluate(inputs)
            acc += result.value
        
        output_signal.append(acc)
    
    print(f"Input Signal:  {input_signal}")
    print(f"Output Signal: {output_signal}")


def main():
    """Main demonstration of FPGA simulator capabilities"""
    print("=== Advanced FPGA Simulator Demo ===")
    print(f"GPU Support: {GPU_AVAILABLE}")
    print(f"Quantum Support: {QUANTUM_AVAILABLE}")
    print(f"ML Support: {ML_AVAILABLE}")
    
    # Create FPGA fabric
    fpga = ParallelFPGAFabric(rows=8, cols=8, use_gpu=True, num_workers=4)
    
    # 1. Configure as 4-bit adder
    print("\n1. Configuring 4-bit Adder...")
    adder_bitstream = create_adder_bitstream(4)
    fpga.configure_from_bitstream(adder_bitstream)
    
    # Test adder
    a = 5  # 0101
    b = 3  # 0011
    carry_in = 0
    
    # Create inputs for each bit position
    for i in range(4):
        bit_a = (a >> i) & 1
        bit_b = (b >> i) & 1
        inputs = np.array([bit_a, bit_b, carry_in, 0])
        
        # Evaluate sum and carry
        sum_signal = fpga.clbs[i][0].evaluate(inputs)
        carry_signal = fpga.clbs[i][1].evaluate(inputs)
        
        print(f"Bit {i}: {bit_a} + {bit_b} + {carry_in} = {sum_signal.value} (carry: {carry_signal.value})")
        carry_in = carry_signal.value
    
    # 2. Performance benchmarking
    benchmark_parallel_vs_sequential(fpga)
    
    # 3. Test quantum features
    test_quantum_enhanced_logic(fpga)
    
    # 4. Neural network accelerator
    print("\n4. Configuring Neural Network Accelerator...")
    nn_bitstream = create_neural_network_accelerator()
    fpga.configure_from_bitstream(nn_bitstream)
    
    # 5. ML routing optimization
    test_ml_routing_optimization(fpga)
    
    # 6. Digital filter
    simulate_digital_filter(fpga)
    
    # 7. Visualize utilization
    visualize_fpga_utilization(fpga)
    
    # 8. Performance metrics
    print("\n=== Final Performance Metrics ===")
    metrics = fpga.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Note: Import these from the main simulator file
    try:
        from fpga_simulator import ParallelFPGAFabric, GPU_AVAILABLE, QUANTUM_AVAILABLE, ML_AVAILABLE
        main()
    except ImportError:
        print("Please save the first artifact as 'fpga_simulator.py' and run this demo.")
        print("\nAlternatively, you can copy both files into one and run directly.")
