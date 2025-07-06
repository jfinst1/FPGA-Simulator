#!/usr/bin/env python3
"""
Basic FPGA Simulator Usage Example

This example demonstrates the fundamental features of the FPGA simulator,
perfect for beginners to understand how to use the various components.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add parent directory to path to import fpga_simulator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fpga_simulator import (AdvancedFPGAFabric, Signal, SignalType,
                               GPU_AVAILABLE, QUANTUM_AVAILABLE, ML_AVAILABLE)
except ImportError:
    print("Error: Could not import fpga_simulator. Make sure it's in the parent directory.")
    sys.exit(1)


def example_1_basic_clb():
    """Example 1: Basic CLB configuration and evaluation"""
    print("\n" + "="*50)
    print("Example 1: Basic CLB Operations")
    print("="*50)
    
    # Create a small FPGA
    fpga = AdvancedFPGAFabric(rows=4, cols=4, num_brams=1, num_dsps=1)
    
    # Get a specific CLB
    clb = fpga.clbs[0][0]
    
    # Configure as AND gate
    # Truth table for 2-input AND (using 4-input LUT)
    # Inputs: A B X X -> Output
    and_truth_table = [
        0, 0, 0, 0,  # 00XX -> 0
        0, 0, 0, 0,  # 01XX -> 0
        0, 0, 0, 0,  # 10XX -> 0
        1, 1, 1, 1   # 11XX -> 1
    ]
    
    clb.configure({'lut': and_truth_table})
    
    # Test the AND gate
    print("\nTesting AND gate:")
    test_inputs = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0]
    ]
    
    for inputs in test_inputs:
        signal = clb.evaluate(np.array(inputs))
        print(f"Inputs: {inputs[:2]} -> Output: {signal.value}")
    
    # Configure as XOR gate
    xor_truth_table = [
        0, 1, 1, 0,  # Pattern for XOR
        1, 0, 0, 1,
        1, 0, 0, 1,
        0, 1, 1, 0
    ]
    
    clb.configure({'lut': xor_truth_table})
    
    print("\nTesting XOR gate:")
    for inputs in test_inputs:
        signal = clb.evaluate(np.array(inputs))
        print(f"Inputs: {inputs[:2]} -> Output: {signal.value}")


def example_2_bram_usage():
    """Example 2: Using Block RAM with ECC"""
    print("\n" + "="*50)
    print("Example 2: Block RAM Usage")
    print("="*50)
    
    # Create FPGA
    fpga = AdvancedFPGAFabric(rows=2, cols=2, num_brams=2, num_dsps=0)
    
    # Get first BRAM
    bram = fpga.brams[0]
    print(f"BRAM specifications:")
    print(f"  Depth: {bram.depth} words")
    print(f"  Width: {bram.width} bits")
    print(f"  ECC enabled: {bram.use_ecc}")
    
    # Write some data
    print("\nWriting data to BRAM...")
    test_data = [0x42, 0xFF, 0xAA, 0x55, 0x12, 0x34, 0x56, 0x78]
    
    for addr, data in enumerate(test_data):
        bram.write(addr, data)
        print(f"  Address {addr}: wrote 0x{data:02X}")
    
    # Read back data
    print("\nReading data from BRAM...")
    for addr in range(len(test_data)):
        data, error_detected = bram.read(addr)
        status = " (error corrected)" if error_detected else ""
        print(f"  Address {addr}: read 0x{data:02X}{status}")
    
    # Demonstrate ECC by injecting an error
    print("\nDemonstrating ECC (Error Correction)...")
    
    # Manually flip a bit in memory (simulate soft error)
    addr = 0
    original = bram.memory[addr, 0]
    bram.memory[addr, 0] ^= 1  # Flip one bit
    
    # Read with ECC
    data, error_detected = bram.read(addr)
    print(f"  Single bit error injected at address {addr}")
    print(f"  Read result: 0x{data:02X}, Error detected: {error_detected}")
    print(f"  Original data: 0x{test_data[addr]:02X}")
    print(f"  ECC successfully corrected!" if data == test_data[addr] else "  ECC failed!")


def example_3_dsp_operations():
    """Example 3: DSP block operations"""
    print("\n" + "="*50)
    print("Example 3: DSP Block Operations")
    print("="*50)
    
    # Create FPGA
    fpga = AdvancedFPGAFabric(rows=2, cols=2, num_brams=0, num_dsps=2)
    
    # Get DSP block
    dsp = fpga.dsps[0]
    
    # Simple multiplication
    print("\nSimple multiplication:")
    a, b = 7, 9
    result = dsp.multiply(a, b)
    print(f"  {a} × {b} = {result}")
    
    # MAC operations
    print("\nMAC (Multiply-Accumulate) operations:")
    dsp.accumulator = 0  # Reset
    
    # Calculate dot product: [1,2,3] · [4,5,6]
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    
    for a, b in zip(vec1, vec2):
        result = dsp.mac(a, b)
        print(f"  MAC({a}, {b}): accumulator = {result}")
    
    print(f"  Dot product result: {dsp.accumulator}")
    
    # Signal processing: Simple FIR filter
    print("\nFIR Filter Example:")
    
    # Generate input signal (square wave + noise)
    t = np.linspace(0, 1, 100)
    signal = np.sign(np.sin(2 * np.pi * 5 * t)) + 0.1 * np.random.randn(100)
    
    # Simple moving average filter coefficients
    coeffs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # 5-tap averaging
    
    # Apply filter
    filtered = dsp.fir_filter(signal, coeffs)
    
    # Plot results
    plt.figure(figsize=(10, 4))
    plt.plot(t[:50], signal[:50], 'b-', alpha=0.5, label='Noisy Input')
    plt.plot(t[:50], filtered[:50], 'r-', linewidth=2, label='Filtered Output')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('DSP FIR Filter Example')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # FFT acceleration
    if len(fpga.dsps) > 0:
        print("\nFFT Acceleration:")
        fft_result = dsp.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/100)
        
        plt.figure(figsize=(8, 4))
        plt.plot(freqs[:50], np.abs(fft_result[:50]))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('FFT of Signal (DSP Accelerated)')
        plt.grid(True, alpha=0.3)
        plt.show()


def example_4_parallel_evaluation():
    """Example 4: Parallel CLB evaluation"""
    print("\n" + "="*50)
    print("Example 4: Parallel Processing")
    print("="*50)
    
    # Create larger FPGA
    fpga = AdvancedFPGAFabric(rows=4, cols=4, num_brams=1, num_dsps=1)
    
    # Configure multiple CLBs
    print("Configuring 4 CLBs with different functions...")
    
    # CLB[0,0]: AND gate
    fpga.clbs[0][0].configure({'lut': [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]})
    
    # CLB[0,1]: OR gate
    fpga.clbs[0][1].configure({'lut': [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]})
    
    # CLB[1,0]: XOR gate
    fpga.clbs[1][0].configure({'lut': [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]})
    
    # CLB[1,1]: NAND gate
    fpga.clbs[1][1].configure({'lut': [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]})
    
    # Prepare parallel inputs
    test_inputs = {
        (0, 0): np.array([1, 1, 0, 0]),  # AND
        (0, 1): np.array([1, 0, 0, 0]),  # OR
        (1, 0): np.array([1, 1, 0, 0]),  # XOR
        (1, 1): np.array([1, 1, 0, 0])   # NAND
    }
    
    # Sequential evaluation
    print("\nSequential evaluation:")
    start_time = time.time()
    seq_results = {}
    for pos, inputs in test_inputs.items():
        result = fpga.clbs[pos[0]][pos[1]].evaluate(inputs)
        seq_results[pos] = result.value
    seq_time = time.time() - start_time
    
    # Parallel evaluation
    print("Parallel evaluation:")
    start_time = time.time()
    par_results = fpga.parallel_evaluate(test_inputs)
    par_time = time.time() - start_time
    
    # Display results
    gates = ['AND', 'OR', 'XOR', 'NAND']
    positions = [(0,0), (0,1), (1,0), (1,1)]
    
    print("\nResults:")
    for gate, pos in zip(gates, positions):
        seq_val = seq_results[pos]
        par_val = par_results[pos].value
        print(f"  {gate} gate: Sequential={seq_val}, Parallel={par_val}")
    
    print(f"\nTiming:")
    print(f"  Sequential time: {seq_time*1000:.3f} ms")
    print(f"  Parallel time: {par_time*1000:.3f} ms")
    print(f"  Speedup: {seq_time/par_time:.2f}x")


def example_5_hdl_compilation():
    """Example 5: HDL to bitstream compilation"""
    print("\n" + "="*50)
    print("Example 5: HDL Compilation")
    print("="*50)
    
    # Create FPGA
    fpga = AdvancedFPGAFabric(rows=4, cols=4, num_brams=1, num_dsps=1)
    
    # Simple Verilog code
    verilog_code = """
    module half_adder(sum, carry, a, b);
        output sum, carry;
        input a, b;
        
        xor g1 (sum, a, b);
        and g2 (carry, a, b);
    endmodule
    """
    
    print("Verilog code:")
    print(verilog_code)
    
    try:
        # Compile to bitstream
        bitstream = fpga.compile_hdl(verilog_code)
        
        print(f"\nCompilation successful!")
        print(f"Bitstream metadata:")
        print(f"  Timestamp: {time.ctime(bitstream['metadata']['timestamp'])}")
        print(f"  Target: {bitstream['metadata']['rows']}x{bitstream['metadata']['cols']} FPGA")
        print(f"  CLBs configured: {len(bitstream['clbs'])}")
        print(f"  Routes: {len(bitstream['routes'])}")
        
        # Configure FPGA with bitstream
        fpga.configure_from_bitstream(bitstream)
        
        print("\nFPGA configured successfully!")
        
    except Exception as e:
        print(f"Compilation error: {e}")


def example_6_noise_and_power():
    """Example 6: Power noise simulation"""
    print("\n" + "="*50)
    print("Example 6: Power and Noise Simulation")
    print("="*50)
    
    # Create FPGA
    fpga = AdvancedFPGAFabric(rows=8, cols=8, num_brams=2, num_dsps=2)
    
    # Simulate power consumption
    print("Simulating power distribution network...")
    
    # Create activity pattern (hot spot in center)
    load_current = np.random.rand(fpga.rows, fpga.cols) * 0.05  # Base load
    
    # Add hot spot
    center_row, center_col = fpga.rows // 2, fpga.cols // 2
    load_current[center_row-1:center_row+2, center_col-1:center_col+2] = 0.5
    
    # Simulate power noise
    fpga.simulate_power_noise(load_current)
    
    # Visualize power grid
    plt.figure(figsize=(10, 8))
    
    # Voltage map
    plt.subplot(2, 2, 1)
    im = plt.imshow(fpga.power_grid, cmap='RdYlGn', vmin=0.9, vmax=1.1)
    plt.colorbar(im, label='Voltage (V)')
    plt.title('Power Grid Voltage Distribution')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Current load
    plt.subplot(2, 2, 2)
    im = plt.imshow(load_current, cmap='hot')
    plt.colorbar(im, label='Current (A)')
    plt.title('Current Load Distribution')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Voltage drop
    plt.subplot(2, 2, 3)
    voltage_drop = 1.0 - fpga.power_grid
    im = plt.imshow(voltage_drop * 1000, cmap='Reds')  # Convert to mV
    plt.colorbar(im, label='Voltage Drop (mV)')
    plt.title('IR Drop')
    plt.xlabel('Column')
    plt.ylabel('Row')
    
    # Statistics
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.8, 'Power Statistics:', fontsize=14, weight='bold')
    plt.text(0.1, 0.6, f'Average voltage: {np.mean(fpga.power_grid):.3f} V')
    plt.text(0.1, 0.5, f'Min voltage: {np.min(fpga.power_grid):.3f} V')
    plt.text(0.1, 0.4, f'Max voltage: {np.max(fpga.power_grid):.3f} V')
    plt.text(0.1, 0.3, f'Voltage std dev: {np.std(fpga.power_grid)*1000:.1f} mV')
    plt.text(0.1, 0.1, f'Total current: {np.sum(load_current):.2f} A')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Demonstrate signal noise
    print("\nSignal noise demonstration:")
    
    # Clean signal
    clean_signal = 1.0
    
    # Add different noise types
    thermal_noise = fpga.noise_model.add_thermal_noise(clean_signal)
    supply_noise = fpga.noise_model.add_supply_noise(clean_signal)
    
    # Crosstalk between adjacent signals
    signals = [1.0, 0.0, 1.0, 0.0, 1.0]
    noisy_signals = fpga.noise_model.add_crosstalk(signals, coupling=0.2)
    
    print(f"  Clean signal: {clean_signal:.3f} V")
    print(f"  With thermal noise: {thermal_noise:.3f} V")
    print(f"  With supply noise: {supply_noise:.3f} V")
    print(f"\nCrosstalk demonstration:")
    print(f"  Original signals: {signals}")
    print(f"  With crosstalk:   {[f'{s:.3f}' for s in noisy_signals]}")


def example_7_fault_tolerance():
    """Example 7: Fault injection and recovery"""
    print("\n" + "="*50)
    print("Example 7: Fault Tolerance")
    print("="*50)
    
    # Create FPGA
    fpga = AdvancedFPGAFabric(rows=4, cols=4, num_brams=2, num_dsps=1)
    
    print("Testing fault tolerance mechanisms...")
    
    # Inject faults and test recovery
    recovery_stats = fpga.inject_and_recover_faults(num_faults=20)
    
    print("\nFault injection results:")
    for key, value in recovery_stats.items():
        print(f"  {key}: {value}")
    
    # Calculate metrics
    if recovery_stats['seu_injected'] > 0:
        recovery_rate = recovery_stats['seu_recovered'] / recovery_stats['seu_injected']
        print(f"\nReliability metrics:")
        print(f"  SEU recovery rate: {recovery_rate:.1%}")
        print(f"  System reliability: {'High' if recovery_rate > 0.8 else 'Medium'}")
    
    # Demonstrate Triple Modular Redundancy (TMR)
    print("\nTriple Modular Redundancy (TMR) demonstration:")
    
    # Configure three CLBs identically (AND gate)
    and_lut = [0]*12 + [1]*4  # AND truth table
    for i in range(3):
        fpga.clbs[0][i].configure({'lut': and_lut})
    
    # Test with fault injection
    test_input = np.array([1, 1, 0, 0])
    results = []
    
    print(f"Input: {test_input[:2]}")
    for i in range(3):
        result = fpga.clbs[0][i].evaluate(test_input)
        
        # Inject fault in one CLB
        if i == 1 and np.random.random() < 0.8:
            result.value ^= 1  # Flip output
            print(f"  CLB {i}: {result.value} (FAULTY)")
        else:
            print(f"  CLB {i}: {result.value}")
        
        results.append(result.value)
    
    # Majority voting
    voted_result = max(set(results), key=results.count)
    print(f"  Voted output: {voted_result} (CORRECT)")


def show_system_info():
    """Display system information and available features"""
    print("\n" + "="*50)
    print("FPGA Simulator System Information")
    print("="*50)
    
    print(f"\nAvailable Features:")
    print(f"  GPU Acceleration: {'✓ Enabled' if GPU_AVAILABLE else '✗ Disabled'}")
    print(f"  Quantum Computing: {'✓ Enabled' if QUANTUM_AVAILABLE else '✗ Disabled'}")
    print(f"  Machine Learning: {'✓ Enabled' if ML_AVAILABLE else '✗ Disabled'}")
    
    if not GPU_AVAILABLE:
        print("\n  To enable GPU: pip install cupy-cuda11x")
    if not QUANTUM_AVAILABLE:
        print("  To enable Quantum: pip install qiskit")
    if not ML_AVAILABLE:
        print("  To enable ML: pip install scikit-learn")
    
    print(f"\nPython version: {sys.version.split()[0]}")
    print(f"NumPy version: {np.__version__}")


def main():
    """Run all examples"""
    print("="*50)
    print("FPGA Simulator - Basic Examples")
    print("="*50)
    
    # Show system info
    show_system_info()
    
    # Run examples
    examples = [
        ("Basic CLB Operations", example_1_basic_clb),
        ("Block RAM Usage", example_2_bram_usage),
        ("DSP Operations", example_3_dsp_operations),
        ("Parallel Processing", example_4_parallel_evaluation),
        ("HDL Compilation", example_5_hdl_compilation),
        ("Power and Noise", example_6_noise_and_power),
        ("Fault Tolerance", example_7_fault_tolerance)
    ]
    
    print("\nSelect an example to run:")
    for i, (name, _) in enumerate(examples):
        print(f"  {i+1}. {name}")
    print(f"  {len(examples)+1}. Run all examples")
    print("  0. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (0-{}): ".format(len(examples)+1))
            choice = int(choice)
            
            if choice == 0:
                print("Exiting...")
                break
            elif 1 <= choice <= len(examples):
                examples[choice-1][1]()
            elif choice == len(examples) + 1:
                for name, func in examples:
                    print(f"\nRunning: {name}")
                    func()
                break
            else:
                print("Invalid choice. Please try again.")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please enter a valid number.")
    
    print("\nThank you for trying the FPGA Simulator!")


if __name__ == "__main__":
    main()
