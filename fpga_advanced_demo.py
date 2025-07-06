import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import List, Dict, Tuple
import time
import json

# Visualization tools for FPGA simulator
class FPGAVisualizer:
    """Advanced visualization for FPGA simulation"""
    
    def __init__(self, fpga: 'AdvancedFPGAFabric'):
        self.fpga = fpga
        self.fig = None
        self.axes = {}
        
    def create_layout_view(self) -> plt.Figure:
        """Visualize FPGA layout with CLBs, BRAMs, and DSPs"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Draw CLBs
        for i in range(self.fpga.rows):
            for j in range(self.fpga.cols):
                # CLB rectangle
                rect = FancyBboxPatch(
                    (j * 1.2, i * 1.2), 1.0, 1.0,
                    boxstyle="round,pad=0.1",
                    facecolor='lightblue',
                    edgecolor='black',
                    linewidth=1
                )
                ax.add_patch(rect)
                ax.text(j * 1.2 + 0.5, i * 1.2 + 0.5, f'CLB\n({i},{j})', 
                       ha='center', va='center', fontsize=8)
        
        # Draw BRAMs
        bram_start_x = self.fpga.cols * 1.2 + 1
        for i, bram in enumerate(self.fpga.brams):
            rect = FancyBboxPatch(
                (bram_start_x, i * 2), 1.5, 1.8,
                boxstyle="round,pad=0.1",
                facecolor='lightgreen',
                edgecolor='darkgreen',
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(bram_start_x + 0.75, i * 2 + 0.9, f'BRAM\n{i}', 
                   ha='center', va='center', fontsize=10, weight='bold')
        
        # Draw DSPs
        dsp_start_y = self.fpga.rows * 1.2 + 1
        for i, dsp in enumerate(self.fpga.dsps):
            rect = FancyBboxPatch(
                (i * 2.5, dsp_start_y), 2.0, 1.5,
                boxstyle="round,pad=0.1",
                facecolor='lightyellow',
                edgecolor='orange',
                linewidth=2
            )
            ax.add_patch(rect)
            ax.text(i * 2.5 + 1.0, dsp_start_y + 0.75, f'DSP\n{i}', 
                   ha='center', va='center', fontsize=10, weight='bold')
        
        # Draw QPU
        qpu_rect = FancyBboxPatch(
            (bram_start_x, self.fpga.rows * 1.2 + 1), 1.5, 1.5,
            boxstyle="round,pad=0.1",
            facecolor='lavender',
            edgecolor='purple',
            linewidth=3
        )
        ax.add_patch(qpu_rect)
        ax.text(bram_start_x + 0.75, self.fpga.rows * 1.2 + 1.75, 'QPU', 
               ha='center', va='center', fontsize=12, weight='bold', color='purple')
        
        ax.set_xlim(-0.5, bram_start_x + 2)
        ax.set_ylim(-0.5, dsp_start_y + 2)
        ax.set_aspect('equal')
        ax.set_title('FPGA Architecture Layout', fontsize=16, weight='bold')
        ax.axis('off')
        
        return fig
    
    def visualize_routing(self, routing: Dict) -> plt.Figure:
        """Visualize routing paths"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Draw CLBs first
        for i in range(self.fpga.rows):
            for j in range(self.fpga.cols):
                rect = Rectangle((j * 1.2, i * 1.2), 1.0, 1.0,
                               facecolor='lightgray', edgecolor='black')
                ax.add_patch(rect)
        
        # Draw routing paths
        colors = plt.cm.rainbow(np.linspace(0, 1, len(routing)))
        
        for idx, (net, path) in enumerate(routing.items()):
            segments = []
            for segment in path:
                src, dst = segment
                segments.append([
                    (src[1] * 1.2 + 0.5, src[0] * 1.2 + 0.5),
                    (dst[1] * 1.2 + 0.5, dst[0] * 1.2 + 0.5)
                ])
            
            lc = LineCollection(segments, colors=[colors[idx]], linewidths=2)
            ax.add_collection(lc)
            
            # Label the net
            if segments:
                ax.text(segments[0][0][0], segments[0][0][1], net.split('->')[0], 
                       fontsize=8, weight='bold')
        
        ax.set_xlim(-0.5, self.fpga.cols * 1.2)
        ax.set_ylim(-0.5, self.fpga.rows * 1.2)
        ax.set_aspect('equal')
        ax.set_title('FPGA Routing Visualization', fontsize=16, weight='bold')
        ax.invert_yaxis()
        
        return fig
    
    def plot_signal_timing(self, signals: List[Signal], time_window: float = 100e-9) -> plt.Figure:
        """Plot signal timing diagram with noise"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        # Extract timing data
        times = [s.timestamp for s in signals]
        values = [s.value if isinstance(s.value, (int, float)) else s.value[0] for s in signals]
        noise_levels = [s.noise_level for s in signals]
        
        # Plot digital signals
        ax1.step(times, values, where='post', linewidth=2, color='blue', label='Digital Value')
        ax1.fill_between(times, 0, values, step='post', alpha=0.3)
        ax1.set_ylabel('Digital Value', fontsize=12)
        ax1.set_ylim(-0.1, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot analog representation with noise
        analog_values = np.array(values) + np.array(noise_levels)
        ax2.plot(times, analog_values, linewidth=1, color='red', alpha=0.7, label='With Noise')
        ax2.plot(times, values, linewidth=2, color='blue', label='Ideal')
        ax2.set_ylabel('Analog Level', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot noise spectrum
        if len(noise_levels) > 10:
            noise_fft = np.fft.fft(noise_levels)
            freqs = np.fft.fftfreq(len(noise_levels), d=(times[1] - times[0]) if len(times) > 1 else 1e-9)
            ax3.semilogy(freqs[:len(freqs)//2], np.abs(noise_fft[:len(freqs)//2]), 
                        linewidth=1, color='green')
            ax3.set_ylabel('Noise Spectrum', fontsize=12)
            ax3.set_xlabel('Frequency (Hz)', fontsize=12)
            ax3.grid(True, alpha=0.3)
        
        fig.suptitle('Signal Timing Analysis with Noise', fontsize=16, weight='bold')
        plt.tight_layout()
        
        return fig
    
    def visualize_power_grid(self) -> plt.Figure:
        """Visualize power distribution network with noise"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Voltage distribution
        im1 = ax1.imshow(self.fpga.power_grid, cmap='RdYlGn', vmin=0.9, vmax=1.1)
        ax1.set_title('Power Grid Voltage (V)', fontsize=14, weight='bold')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        plt.colorbar(im1, ax=ax1)
        
        # Add contour lines for voltage drops
        contours = ax1.contour(self.fpga.power_grid, levels=[0.95, 1.0, 1.05], 
                              colors='black', linewidths=1, alpha=0.5)
        ax1.clabel(contours, inline=True, fontsize=8)
        
        # Noise histogram
        noise_data = (self.fpga.power_grid - 1.0).flatten() * 1000  # Convert to mV
        ax2.hist(noise_data, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_xlabel('Voltage Noise (mV)', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title('Power Supply Noise Distribution', fontsize=14, weight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Fit and plot Gaussian
        from scipy.stats import norm
        mu, std = norm.fit(noise_data)
        x = np.linspace(noise_data.min(), noise_data.max(), 100)
        ax2.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2, 
                label=f'Gaussian fit\nμ={mu:.1f}mV, σ={std:.1f}mV')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def animate_quantum_state(self, num_frames: int = 100) -> animation.FuncAnimation:
        """Animate quantum state evolution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Initialize quantum circuit
        if QUANTUM_AVAILABLE:
            qc = QuantumCircuit(3)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(1, 2)
            
            backend = Aer.get_backend('statevector_simulator')
            
            # Prepare data for animation
            angles = np.linspace(0, 2*np.pi, num_frames)
            
            def update(frame):
                ax1.clear()
                ax2.clear()
                
                # Add rotation
                qc_frame = qc.copy()
                qc_frame.ry(angles[frame], 0)
                
                # Get statevector
                job = execute(qc_frame, backend)
                result = job.result()
                statevector = result.get_statevector()
                
                # Plot probability amplitudes
                probs = np.abs(statevector)**2
                ax1.bar(range(len(probs)), probs, color='purple', alpha=0.7)
                ax1.set_xlabel('Basis State')
                ax1.set_ylabel('Probability')
                ax1.set_title(f'Quantum State Probabilities (t={frame})')
                ax1.set_ylim(0, 1)
                
                # Plot phase
                phases = np.angle(statevector)
                ax2.scatter(np.real(statevector), np.imag(statevector), 
                          c=phases, cmap='hsv', s=100)
                ax2.set_xlabel('Real')
                ax2.set_ylabel('Imaginary')
                ax2.set_title('Quantum State Phase Space')
                ax2.set_xlim(-1, 1)
                ax2.set_ylim(-1, 1)
                ax2.grid(True, alpha=0.3)
                
                # Add unit circle
                circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
                ax2.add_patch(circle)
            
            anim = animation.FuncAnimation(fig, update, frames=num_frames, 
                                         interval=50, blit=False)
            return anim
        else:
            ax1.text(0.5, 0.5, 'Quantum features not available', 
                    transform=ax1.transAxes, ha='center')
            return None


def demo_hdl_compilation():
    """Demonstrate HDL to bitstream compilation"""
    print("\n=== HDL Compilation Demo ===")
    
    # Sample Verilog code
    verilog_code = """
    module full_adder(sum, cout, a, b, cin);
        output sum, cout;
        input a, b, cin;
        
        wire s1, c1, c2;
        
        xor g1 (s1, a, b);
        xor g2 (sum, s1, cin);
        and g3 (c1, a, b);
        and g4 (c2, s1, cin);
        or g5 (cout, c1, c2);
    endmodule
    
    module ripple_adder(sum, cout, a, b, cin);
        output [3:0] sum;
        output cout;
        input [3:0] a, b;
        input cin;
        
        wire c1, c2, c3;
        
        full_adder fa0 (sum[0], c1, a[0], b[0], cin);
        full_adder fa1 (sum[1], c2, a[1], b[1], c1);
        full_adder fa2 (sum[2], c3, a[2], b[2], c2);
        full_adder fa3 (sum[3], cout, a[3], b[3], c3);
    endmodule
    """
    
    # Create FPGA
    fpga = AdvancedFPGAFabric(rows=8, cols=8, num_brams=4, num_dsps=2)
    
    # Compile HDL
    bitstream = fpga.compile_hdl(verilog_code)
    
    print(f"Generated bitstream with {len(bitstream['clbs'])} CLBs configured")
    print(f"Routing uses {len(bitstream['routes'])} connections")
    
    return fpga, bitstream


def demo_dsp_operations():
    """Demonstrate DSP block operations"""
    print("\n=== DSP Operations Demo ===")
    
    fpga = AdvancedFPGAFabric(rows=4, cols=4, num_dsps=4)
    
    # Generate test signal
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
    noise = 0.2 * np.random.randn(len(t))
    noisy_signal = signal + noise
    
    # Design low-pass filter
    cutoff = 80  # Hz
    nyquist = fs / 2
    taps = 31
    h = np.sinc(2 * cutoff / fs * (np.arange(taps) - (taps - 1) / 2))
    h *= np.blackman(taps)
    h /= np.sum(h)
    
    # Filter using DSP blocks
    filtered = fpga.dsp_accelerated_convolution(noisy_signal, h)
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    ax1.plot(t[:200], noisy_signal[:200], 'b-', alpha=0.5, label='Noisy')
    ax1.plot(t[:200], signal[:200], 'r-', linewidth=2, label='Original')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.set_title('Input Signal')
    
    ax2.plot(t[:200], filtered[:200], 'g-', linewidth=2, label='Filtered')
    ax2.plot(t[:200], signal[:200], 'r--', alpha=0.5, label='Original')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.set_title('DSP Filtered Output')
    
    # Frequency response
    freq = np.fft.fftfreq(len(noisy_signal), 1/fs)
    fft_noisy = np.fft.fft(noisy_signal)
    fft_filtered = np.fft.fft(filtered)
    
    ax3.semilogy(freq[:len(freq)//2], np.abs(fft_noisy[:len(freq)//2]), 
                 'b-', alpha=0.5, label='Noisy')
    ax3.semilogy(freq[:len(freq)//2], np.abs(fft_filtered[:len(freq)//2]), 
                 'g-', linewidth=2, label='Filtered')
    ax3.axvline(cutoff, color='r', linestyle='--', label='Cutoff')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.legend()
    ax3.set_title('Frequency Response')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fpga


def demo_quantum_algorithms():
    """Demonstrate quantum algorithm acceleration"""
    print("\n=== Quantum Algorithm Demo ===")
    
    if not QUANTUM_AVAILABLE:
        print("Quantum features not available. Install qiskit to enable.")
        return
    
    fpga = AdvancedFPGAFabric(rows=4, cols=4)
    
    # 1. Quantum Fourier Transform
    print("\n1. Quantum Fourier Transform:")
    input_state = [1, 0, 1, 0, 1, 0, 0, 0]  # Binary: 10100000
    qft_result = fpga.qpu.quantum_fourier_transform(input_state)
    print(f"Input: {input_state}")
    print(f"QFT output (first 8 amplitudes): {qft_result[:8]}")
    
    # 2. Grover's Search
    print("\n2. Grover's Search Algorithm:")
    # Search for item satisfying condition
    def search_condition(x):
        return x == 13  # Looking for 13 in 4-bit space
    
    result = fpga.run_quantum_accelerated_search(16, search_condition)
    print(f"Found item: {result}")
    
    # 3. Variational Quantum Eigensolver
    print("\n3. VQE for Ground State Energy:")
    # Simple 2x2 Hamiltonian
    H = np.array([[1, 0.5], [0.5, -1]])
    ground_energy = fpga.qpu.variational_quantum_eigensolver(H)
    classical_ground = np.min(np.linalg.eigvals(H))
    print(f"VQE result: {ground_energy:.4f}")
    print(f"Classical result: {classical_ground:.4f}")
    
    return fpga


def demo_fault_tolerance():
    """Demonstrate fault tolerance and recovery"""
    print("\n=== Fault Tolerance Demo ===")
    
    fpga = AdvancedFPGAFabric(rows=4, cols=4, num_brams=4)
    
    # Test BRAM with ECC
    print("\n1. BRAM ECC Testing:")
    bram = fpga.brams[0]
    
    # Write test data
    test_data = [0x55, 0xAA, 0xFF, 0x00, 0x12, 0x34, 0x56, 0x78]
    for addr, data in enumerate(test_data):
        bram.write(addr, data)
    
    # Inject faults and read back
    errors_detected = 0
    errors_corrected = 0
    
    for addr in range(len(test_data)):
        # Corrupt memory randomly
        if np.random.random() < 0.3:
            bit_pos = np.random.randint(0, bram.width)
            bram.memory[addr, bit_pos] ^= 1  # Flip bit
        
        read_data, error = bram.read(addr)
        if error:
            errors_detected += 1
            if read_data == test_data[addr]:
                errors_corrected += 1
    
    print(f"Errors detected: {errors_detected}")
    print(f"Errors corrected: {errors_corrected}")
    
    # Test system-wide fault injection
    print("\n2. System-wide Fault Injection:")
    recovery_stats = fpga.inject_and_recover_faults(num_faults=50)
    
    print("Fault Recovery Statistics:")
    for key, value in recovery_stats.items():
        print(f"  {key}: {value}")
    
    # Calculate reliability metrics
    if recovery_stats['seu_injected'] > 0:
        seu_recovery_rate = recovery_stats['seu_recovered'] / recovery_stats['seu_injected']
        print(f"\nSEU Recovery Rate: {seu_recovery_rate:.2%}")
    
    return fpga


def demo_noise_analysis():
    """Demonstrate power noise analysis"""
    print("\n=== Noise Analysis Demo ===")
    
    fpga = AdvancedFPGAFabric(rows=8, cols=8)
    
    # Simulate varying load conditions
    load_current = np.random.rand(fpga.rows, fpga.cols) * 0.1  # 0-100mA
    
    # Add hotspots
    load_current[2:4, 2:4] = 0.5  # High activity region
    
    # Simulate power noise
    fpga.simulate_power_noise(load_current)
    
    # Create visualizer
    viz = FPGAVisualizer(fpga)
    
    # Visualize power grid
    fig = viz.visualize_power_grid()
    plt.show()
    
    # Analyze crosstalk
    print("\nCrosstalk Analysis:")
    signals = [1.0, 0.0, 1.0, 0.0, 1.0]
    noisy_signals = fpga.noise_model.add_crosstalk(signals, coupling=0.15)
    
    print(f"Original signals: {signals}")
    print(f"With crosstalk:   {[f'{s:.3f}' for s in noisy_signals]}")
    
    return fpga


def create_comprehensive_report(fpga: 'AdvancedFPGAFabric'):
    """Generate comprehensive FPGA analysis report"""
    print("\n=== Generating Comprehensive Report ===")
    
    report = {
        'timestamp': time.time(),
        'configuration': {
            'rows': fpga.rows,
            'cols': fpga.cols,
            'num_brams': len(fpga.brams),
            'num_dsps': len(fpga.dsps),
            'gpu_enabled': fpga.use_gpu and GPU_AVAILABLE,
            'quantum_enabled': QUANTUM_AVAILABLE
        },
        'performance': fpga.get_performance_metrics(),
        'noise_analysis': {
            'avg_voltage': np.mean(fpga.power_grid),
            'voltage_std': np.std(fpga.power_grid),
            'min_voltage': np.min(fpga.power_grid),
            'max_voltage': np.max(fpga.power_grid)
        },
        'fault_tolerance': {
            'bram_ecc_enabled': all(bram.use_ecc for bram in fpga.brams),
            'seu_rate': fpga.fault_model.seu_rate,
            'fault_history_count': len(fpga.fault_model.fault_history)
        }
    }
    
    # Save report
    with open('fpga_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=float)
    
    print("Report saved to fpga_analysis_report.json")
    
    # Create summary visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Resource utilization pie chart
    resources = ['CLBs', 'BRAMs', 'DSPs', 'QPU']
    sizes = [fpga.rows * fpga.cols, len(fpga.brams), len(fpga.dsps), 1]
    ax1.pie(sizes, labels=resources, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Resource Distribution')
    
    # Performance metrics bar chart
    metrics = list(report['performance'].keys())[:4]
    values = [report['performance'][m] for m in metrics]
    ax2.bar(metrics, values)
    ax2.set_title('Performance Metrics')
    ax2.tick_params(axis='x', rotation=45)
    
    # Voltage distribution
    ax3.hist(fpga.power_grid.flatten(), bins=30, density=True, alpha=0.7)
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Density')
    ax3.set_title('Power Grid Voltage Distribution')
    ax3.axvline(1.0, color='r', linestyle='--', label='Nominal')
    ax3.legend()
    
    # Feature availability
    features = ['GPU', 'Quantum', 'ML', 'ECC']
    available = [GPU_AVAILABLE, QUANTUM_AVAILABLE, ML_AVAILABLE, 
                 all(bram.use_ecc for bram in fpga.brams)]
    colors = ['green' if a else 'red' for a in available]
    ax4.bar(features, [1 if a else 0 for a in available], color=colors)
    ax4.set_ylim(0, 1.2)
    ax4.set_ylabel('Available')
    ax4.set_title('Feature Availability')
    
    plt.tight_layout()
    plt.savefig('fpga_summary_report.png', dpi=300)
    plt.show()
    
    return report


def main():
    """Main demonstration of all FPGA features"""
    print("=== Advanced FPGA Simulator - Complete Demo ===")
    print(f"System Status:")
    print(f"  GPU Support: {'✓' if GPU_AVAILABLE else '✗'}")
    print(f"  Quantum Support: {'✓' if QUANTUM_AVAILABLE else '✗'}")
    print(f"  ML Support: {'✓' if ML_AVAILABLE else '✗'}")
    
    # 1. HDL Compilation Demo
    fpga, bitstream = demo_hdl_compilation()
    
    # 2. Visualize architecture
    viz = FPGAVisualizer(fpga)
    layout_fig = viz.create_layout_view()
    plt.show()
    
    # 3. DSP Operations
    fpga_dsp = demo_dsp_operations()
    
    # 4. Quantum Algorithms
    if QUANTUM_AVAILABLE:
        fpga_quantum = demo_quantum_algorithms()
    
    # 5. Fault Tolerance
    fpga_fault = demo_fault_tolerance()
    
    # 6. Noise Analysis
    fpga_noise = demo_noise_analysis()
    
    # 7. Generate comprehensive report
    report = create_comprehensive_report(fpga_noise)
    
    print("\n=== Demo Complete ===")
    print("Generated files:")
    print("  - fpga_utilization.png")
    print("  - fpga_analysis_report.json")
    print("  - fpga_summary_report.png")
    
    # 8. Real-time signal timing visualization
    print("\nGenerating signal timing visualization...")
    
    # Generate test signals with noise
    test_signals = []
    for i in range(100):
        sig = fpga.clbs[0][0].evaluate(np.random.randint(0, 2, 4))
        test_signals.append(sig)
    
    timing_fig = viz.plot_signal_timing(test_signals)
    plt.show()
    
    # 9. Routing visualization
    if 'routing' in locals() and routing:
        routing_fig = viz.visualize_routing(routing)
        plt.show()


if __name__ == "__main__":
    # Import the main FPGA simulator
    try:
        from fpga_simulator import (AdvancedFPGAFabric, GPU_AVAILABLE, 
                                   QUANTUM_AVAILABLE, ML_AVAILABLE,
                                   Signal, QuantumCircuit, Aer, execute)
        main()
    except ImportError:
        print("Note: This demo requires the main fpga_simulator module.")
        print("For standalone testing, you can combine both files.")
        
        # Run minimal demo
        print("\nRunning minimal architecture visualization demo...")
        
        # Create mock FPGA for visualization
        class MockFPGA:
            def __init__(self):
                self.rows = 4
                self.cols = 4
                self.brams = [None] * 2
                self.dsps = [None] * 2
                self.power_grid = np.random.normal(1.0, 0.05, (4, 4))
        
        mock_fpga = MockFPGA()
        viz = FPGAVisualizer(mock_fpga)
        fig = viz.create_layout_view()
        plt.show()
