#!/usr/bin/env python3
"""
Quantum Algorithm Examples for FPGA Simulator

This example demonstrates various quantum algorithms implemented on the FPGA's
Quantum Processing Unit (QPU), including Grover's search, QFT, and VQE.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple
import time
import sys
import os

# Add parent directory to path to import fpga_simulator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fpga_simulator import AdvancedFPGAFabric
    # Check if quantum features are available
    from fpga_simulator import QUANTUM_AVAILABLE
    if QUANTUM_AVAILABLE:
        from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
        from qiskit.visualization import plot_histogram, plot_bloch_multivector
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure fpga_simulator.py is in the parent directory and qiskit is installed.")
    sys.exit(1)


class QuantumSearchDemo:
    """Demonstrations of quantum search algorithms on FPGA"""
    
    def __init__(self, fpga: AdvancedFPGAFabric):
        self.fpga = fpga
        self.qpu = fpga.qpu
        
    def demo_grover_search(self):
        """Demonstrate Grover's algorithm for database search"""
        print("\n=== Grover's Search Algorithm Demo ===")
        
        if not QUANTUM_AVAILABLE:
            print("Quantum features not available. Using classical simulation.")
            return self._classical_grover_demo()
        
        # Different search scenarios
        scenarios = [
            {
                'name': 'Find single item in 4-bit space',
                'size': 16,
                'target': 10,
                'qubits': 4
            },
            {
                'name': 'Find prime number in 5-bit space',
                'size': 32,
                'target': lambda x: self._is_prime(x),
                'qubits': 5
            },
            {
                'name': 'Find pattern in 3-bit space',
                'size': 8,
                'target': lambda x: bin(x).count('1') == 2,  # Two 1s
                'qubits': 3
            }
        ]
        
        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            print(f"Search space size: {scenario['size']}")
            
            # Set QPU qubits
            self.qpu.num_qubits = scenario['qubits']
            
            # Create oracle function
            if isinstance(scenario['target'], int):
                oracle = lambda x: x == scenario['target']
                expected = [scenario['target']]
            else:
                oracle = scenario['target']
                expected = [x for x in range(scenario['size']) if oracle(x)]
            
            print(f"Expected results: {expected}")
            
            # Run Grover's algorithm
            start_time = time.time()
            result = self.fpga.run_quantum_accelerated_search(
                scenario['size'], oracle
            )
            quantum_time = time.time() - start_time
            
            print(f"Grover's result: {result}")
            print(f"Correct: {oracle(result)}")
            print(f"Quantum time: {quantum_time*1000:.2f} ms")
            
            # Compare with classical search
            start_time = time.time()
            classical_result = self._classical_search(scenario['size'], oracle)
            classical_time = time.time() - start_time
            
            print(f"Classical time: {classical_time*1000:.2f} ms")
            print(f"Quantum speedup: {classical_time/quantum_time:.2f}x")
            
            # Visualize probability distribution
            self._visualize_grover_probabilities(scenario['qubits'], expected)
    
    def demo_quantum_fourier_transform(self):
        """Demonstrate Quantum Fourier Transform"""
        print("\n=== Quantum Fourier Transform Demo ===")
        
        if not QUANTUM_AVAILABLE:
            print("Quantum features not available. Using classical FFT.")
            return self._classical_qft_demo()
        
        # Test cases
        test_cases = [
            {
                'name': 'Periodic signal',
                'input': [1, 0, 1, 0, 1, 0, 1, 0],
                'description': 'Alternating pattern'
            },
            {
                'name': 'Single frequency',
                'input': [0, 1, 0, 0, 0, 0, 0, 0],
                'description': 'Delta function at position 1'
            },
            {
                'name': 'Superposition',
                'input': [1, 1, 0, 0, 1, 1, 0, 0],
                'description': 'Mixed frequencies'
            }
        ]
        
        for test in test_cases:
            print(f"\n{test['name']} - {test['description']}:")
            print(f"Input state: {test['input']}")
            
            # Run QFT on FPGA
            start_time = time.time()
            qft_result = self.qpu.quantum_fourier_transform(test['input'])
            qft_time = time.time() - start_time
            
            # Compare with classical FFT
            start_time = time.time()
            fft_result = np.fft.fft(test['input']) / np.sqrt(len(test['input']))
            fft_time = time.time() - start_time
            
            print(f"QFT time: {qft_time*1000:.2f} ms")
            print(f"FFT time: {fft_time*1000:.2f} ms")
            
            # Visualize results
            self._visualize_qft_results(test['input'], qft_result, fft_result)
    
    def demo_variational_quantum_eigensolver(self):
        """Demonstrate VQE for finding ground state energy"""
        print("\n=== Variational Quantum Eigensolver Demo ===")
        
        if not QUANTUM_AVAILABLE:
            print("Quantum features not available. Using classical eigensolvers.")
            return self._classical_vqe_demo()
        
        # Test Hamiltonians
        hamiltonians = [
            {
                'name': 'Simple 2x2 system',
                'H': np.array([[1, 0.5], [0.5, -1]]),
                'qubits': 1
            },
            {
                'name': 'Hydrogen molecule',
                'H': self._get_h2_hamiltonian(),
                'qubits': 2
            },
            {
                'name': 'Heisenberg model (3 spins)',
                'H': self._get_heisenberg_hamiltonian(3),
                'qubits': 3
            }
        ]
        
        for system in hamiltonians:
            print(f"\n{system['name']}:")
            
            # Ensure Hamiltonian is the right size
            H = system['H']
            expected_size = 2 ** system['qubits']
            if H.shape[0] != expected_size:
                print(f"Skipping - Hamiltonian size mismatch")
                continue
            
            # Set QPU qubits
            self.qpu.num_qubits = system['qubits']
            
            # Run VQE
            start_time = time.time()
            vqe_energy = self.qpu.variational_quantum_eigensolver(H)
            vqe_time = time.time() - start_time
            
            # Classical solution
            start_time = time.time()
            eigenvalues = np.linalg.eigvals(H)
            classical_ground = np.min(eigenvalues.real)
            classical_time = time.time() - start_time
            
            print(f"VQE ground state energy: {vqe_energy:.6f}")
            print(f"Classical ground state: {classical_ground:.6f}")
            print(f"Error: {abs(vqe_energy - classical_ground):.6f}")
            print(f"VQE time: {vqe_time*1000:.2f} ms")
            print(f"Classical time: {classical_time*1000:.2f} ms")
            
            # Visualize energy landscape
            self._visualize_energy_landscape(H, vqe_energy, classical_ground)
    
    def demo_quantum_phase_estimation(self):
        """Demonstrate Quantum Phase Estimation"""
        print("\n=== Quantum Phase Estimation Demo ===")
        
        # Create unitary operators with known eigenvalues
        test_cases = [
            {
                'name': 'T gate',
                'unitary': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]),
                'expected_phase': 1/8  # π/4 / 2π
            },
            {
                'name': 'S gate',
                'unitary': np.array([[1, 0], [0, 1j]]),
                'expected_phase': 1/4  # π/2 / 2π
            },
            {
                'name': 'Custom rotation',
                'unitary': self._rotation_matrix(np.pi / 3),
                'expected_phase': 1/6  # π/3 / 2π
            }
        ]
        
        for test in test_cases:
            print(f"\n{test['name']}:")
            print(f"Expected phase: {test['expected_phase']} ({test['expected_phase'] * 2 * np.pi:.3f} radians)")
            
            # Simulate phase estimation (simplified)
            estimated_phase = self._estimate_phase(test['unitary'])
            print(f"Estimated phase: {estimated_phase:.3f}")
            print(f"Error: {abs(estimated_phase - test['expected_phase']):.3f}")
    
    def demo_quantum_machine_learning(self):
        """Demonstrate quantum-enhanced machine learning"""
        print("\n=== Quantum Machine Learning Demo ===")
        
        # Simple quantum kernel for classification
        print("\nQuantum Kernel Support Vector Machine:")
        
        # Generate simple 2D dataset
        np.random.seed(42)
        n_samples = 20
        
        # Class 0: centered at (-1, -1)
        class0 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([-1, -1])
        # Class 1: centered at (1, 1)
        class1 = np.random.randn(n_samples//2, 2) * 0.5 + np.array([1, 1])
        
        X = np.vstack([class0, class1])
        y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
        
        # Compute quantum kernel
        print("Computing quantum kernel matrix...")
        K = self._quantum_kernel(X)
        
        # Visualize
        self._visualize_quantum_classification(X, y, K)
    
    # Helper methods
    def _is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        for i in range(2, int(np.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def _classical_search(self, size: int, oracle: Callable) -> int:
        """Classical linear search"""
        for i in range(size):
            if oracle(i):
                return i
        return -1
    
    def _get_h2_hamiltonian(self) -> np.ndarray:
        """Get simplified H2 molecule Hamiltonian"""
        # Simplified 2-qubit Hamiltonian for H2
        return np.array([
            [0.5, 0, 0, 0.2],
            [0, -0.5, 0.1, 0],
            [0, 0.1, -0.5, 0],
            [0.2, 0, 0, 0.5]
        ])
    
    def _get_heisenberg_hamiltonian(self, n_spins: int) -> np.ndarray:
        """Get Heisenberg model Hamiltonian"""
        # Simplified version for demonstration
        dim = 2 ** n_spins
        H = np.zeros((dim, dim))
        
        # Add nearest-neighbor interactions
        for i in range(n_spins - 1):
            # Simplified interaction terms
            for state in range(dim):
                H[state, state] += 0.5 * ((-1) ** ((state >> i) & 1))
                
        return H
    
    def _rotation_matrix(self, theta: float) -> np.ndarray:
        """Create 2x2 rotation matrix"""
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
    
    def _estimate_phase(self, unitary: np.ndarray) -> float:
        """Simplified phase estimation"""
        eigenvalues = np.linalg.eigvals(unitary)
        phases = np.angle(eigenvalues) / (2 * np.pi)
        return phases[0] % 1
    
    def _quantum_kernel(self, X: np.ndarray) -> np.ndarray:
        """Compute quantum kernel matrix (simplified)"""
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                # Simplified quantum kernel using feature map
                diff = X[i] - X[j]
                K[i, j] = np.exp(-np.linalg.norm(diff)**2)
                
        return K
    
    # Visualization methods
    def _visualize_grover_probabilities(self, n_qubits: int, marked_states: List[int]):
        """Visualize probability distribution during Grover's algorithm"""
        if not QUANTUM_AVAILABLE:
            return
            
        iterations = int(np.pi/4 * np.sqrt(2**n_qubits))
        n_states = 2**n_qubits
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        # Show probability distribution at different iterations
        for idx, iteration in enumerate([0, iterations//3, 2*iterations//3, iterations]):
            probs = self._simulate_grover_iteration(n_qubits, marked_states, iteration)
            
            ax = axes[idx]
            ax.bar(range(n_states), probs, color=['red' if i in marked_states else 'blue' 
                                                  for i in range(n_states)])
            ax.set_title(f'Iteration {iteration}')
            ax.set_xlabel('State')
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            
        plt.tight_layout()
        plt.suptitle("Grover's Algorithm Probability Evolution", y=1.02)
        plt.show()
    
    def _simulate_grover_iteration(self, n_qubits: int, marked_states: List[int], 
                                  iteration: int) -> np.ndarray:
        """Simulate probability distribution after given iterations"""
        n_states = 2**n_qubits
        n_marked = len(marked_states)
        
        # Analytical formula for Grover's algorithm
        theta = np.arcsin(np.sqrt(n_marked / n_states))
        
        # Probability of marked state
        p_marked = np.sin((2 * iteration + 1) * theta) ** 2
        # Probability of unmarked state
        p_unmarked = np.cos((2 * iteration + 1) * theta) ** 2 / (n_states - n_marked)
        
        probs = np.zeros(n_states)
        for i in range(n_states):
            if i in marked_states:
                probs[i] = p_marked / n_marked
            else:
                probs[i] = p_unmarked
                
        return probs
    
    def _visualize_qft_results(self, input_state: List[int], 
                              qft_result: np.ndarray, 
                              fft_result: np.ndarray):
        """Visualize QFT vs FFT results"""
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Input state
        axes[0, 0].stem(input_state)
        axes[0, 0].set_title('Input State')
        axes[0, 0].set_xlabel('Basis State')
        axes[0, 0].set_ylabel('Amplitude')
        
        # QFT magnitude
        axes[0, 1].stem(np.abs(qft_result))
        axes[0, 1].set_title('QFT Magnitude')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('|Amplitude|')
        
        # QFT phase
        axes[0, 2].stem(np.angle(qft_result))
        axes[0, 2].set_title('QFT Phase')
        axes[0, 2].set_xlabel('Frequency')
        axes[0, 2].set_ylabel('Phase (rad)')
        
        # FFT comparison
        axes[1, 0].stem(np.abs(fft_result))
        axes[1, 0].set_title('Classical FFT Magnitude')
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_ylabel('|Amplitude|')
        
        # Difference
        diff = np.abs(qft_result) - np.abs(fft_result)
        axes[1, 1].stem(diff)
        axes[1, 1].set_title('Magnitude Difference (QFT - FFT)')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('Difference')
        
        # Hide last subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _visualize_energy_landscape(self, H: np.ndarray, vqe_energy: float, 
                                   classical_energy: float):
        """Visualize energy landscape for VQE"""
        # For visualization, show energy as function of a parameter
        thetas = np.linspace(0, 2*np.pi, 100)
        energies = []
        
        for theta in thetas:
            # Simple ansatz: rotation around y-axis
            U = np.array([[np.cos(theta/2), -np.sin(theta/2)],
                         [np.sin(theta/2), np.cos(theta/2)]])
            
            if H.shape[0] == 2:
                # Direct calculation for 1-qubit
                psi = U @ np.array([1, 0])  # |0⟩ state
                energy = np.real(np.conj(psi) @ H @ psi)
            else:
                # Simplified for multi-qubit
                energy = np.real(np.trace(H)) / H.shape[0] + 0.5 * np.cos(theta)
                
            energies.append(energy)
        
        plt.figure(figsize=(8, 6))
        plt.plot(thetas, energies, 'b-', label='Energy landscape')
        plt.axhline(classical_energy, color='r', linestyle='--', 
                   label=f'Classical ground: {classical_energy:.3f}')
        plt.axhline(vqe_energy, color='g', linestyle='--', 
                   label=f'VQE result: {vqe_energy:.3f}')
        plt.xlabel('Parameter θ')
        plt.ylabel('Energy')
        plt.title('VQE Energy Landscape')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def _visualize_quantum_classification(self, X: np.ndarray, y: np.ndarray, 
                                        K: np.ndarray):
        """Visualize quantum kernel classification"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot dataset
        colors = ['blue', 'red']
        for i in range(2):
            mask = y == i
            ax1.scatter(X[mask, 0], X[mask, 1], c=colors[i], 
                       label=f'Class {i}', s=100, alpha=0.7)
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_title('Dataset')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot kernel matrix
        im = ax2.imshow(K, cmap='viridis', aspect='auto')
        ax2.set_title('Quantum Kernel Matrix')
        ax2.set_xlabel('Sample index')
        ax2.set_ylabel('Sample index')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.show()
    
    # Classical fallback methods
    def _classical_grover_demo(self):
        """Classical simulation of Grover's algorithm concepts"""
        print("\nRunning classical simulation of Grover's algorithm...")
        
        # Demonstrate the quadratic speedup concept
        sizes = [16, 64, 256]
        for size in sizes:
            classical_steps = size  # Linear search
            quantum_steps = int(np.pi/4 * np.sqrt(size))  # Grover's
            
            print(f"\nSearch space size: {size}")
            print(f"Classical steps needed: {classical_steps}")
            print(f"Quantum steps needed: {quantum_steps}")
            print(f"Speedup: {classical_steps/quantum_steps:.2f}x")
    
    def _classical_qft_demo(self):
        """Classical demonstration of QFT concepts"""
        print("\nRunning classical FFT demonstration...")
        
        # Show how QFT relates to classical FFT
        signal = [1, 0, 1, 0, 1, 0, 1, 0]
        fft_result = np.fft.fft(signal)
        
        print(f"Input signal: {signal}")
        print(f"FFT result magnitudes: {np.abs(fft_result).round(2)}")
        print("Peak at frequency 4 indicates period of 2")
    
    def _classical_vqe_demo(self):
        """Classical demonstration of VQE concepts"""
        print("\nRunning classical eigenvalue demonstration...")
        
        # Simple 2x2 example
        H = np.array([[1, 0.5], [0.5, -1]])
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        
        print(f"Hamiltonian:\n{H}")
        print(f"Eigenvalues: {eigenvalues}")
        print(f"Ground state energy: {eigenvalues[0]:.6f}")
        print(f"Ground state vector: {eigenvectors[:, 0]}")


def benchmark_quantum_algorithms(fpga: AdvancedFPGAFabric):
    """Benchmark various quantum algorithms on FPGA"""
    print("\n=== Quantum Algorithm Benchmarks ===")
    
    algorithms = [
        {
            'name': 'Grover Search (16 items)',
            'func': lambda: fpga.run_quantum_accelerated_search(16, lambda x: x == 10)
        },
        {
            'name': 'QFT (8 points)',
            'func': lambda: fpga.qpu.quantum_fourier_transform([1, 0, 1, 0, 1, 0, 1, 0])
        },
        {
            'name': 'VQE (2x2 system)',
            'func': lambda: fpga.qpu.variational_quantum_eigensolver(
                np.array([[1, 0.5], [0.5, -1]])
            )
        }
    ]
    
    print(f"{'Algorithm':<30} {'Time (ms)':<15} {'Result'}")
    print("-" * 60)
    
    for algo in algorithms:
        start = time.time()
        try:
            result = algo['func']()
            elapsed = (time.time() - start) * 1000
            
            if isinstance(result, np.ndarray):
                result_str = f"Array shape: {result.shape}"
            else:
                result_str = str(result)[:20]
                
            print(f"{algo['name']:<30} {elapsed:<15.2f} {result_str}")
        except Exception as e:
            print(f"{algo['name']:<30} {'Error':<15} {str(e)[:20]}")


def visualize_quantum_fpga_integration(fpga: AdvancedFPGAFabric):
    """Visualize how quantum and classical parts integrate"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Resource allocation
    ax1 = axes[0, 0]
    resources = ['CLBs', 'DSPs', 'BRAMs', 'QPU']
    counts = [
        fpga.rows * fpga.cols,
        len(fpga.dsps),
        len(fpga.brams),
        1
    ]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    ax1.pie(counts, labels=resources, colors=colors, autopct='%1.1f%%')
    ax1.set_title('FPGA Resource Distribution')
    
    # 2. Quantum vs Classical speedup
    ax2 = axes[0, 1]
    problem_sizes = [16, 64, 256, 1024]
    classical_times = problem_sizes
    quantum_times = [int(np.pi/4 * np.sqrt(n)) for n in problem_sizes]
    
    x = np.arange(len(problem_sizes))
    width = 0.35
    
    ax2.bar(x - width/2, classical_times, width, label='Classical', color='#e74c3c')
    ax2.bar(x + width/2, quantum_times, width, label='Quantum', color='#9b59b6')
    ax2.set_xlabel('Problem Size')
    ax2.set_ylabel('Steps Required')
    ax2.set_title('Quantum vs Classical Scaling')
    ax2.set_xticks(x)
    ax2.set_xticklabels(problem_sizes)
    ax2.legend()
    ax2.set_yscale('log')
    
    # 3. Quantum state visualization (Bloch sphere projection)
    ax3 = axes[1, 0]
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2*np.pi, 50)
    
    # Create sphere
    X = np.outer(np.sin(theta), np.cos(phi))
    Y = np.outer(np.sin(theta), np.sin(phi))
    Z = np.outer(np.cos(theta), np.ones(np.size(phi)))
    
    ax3.plot_surface(X, Y, Z, alpha=0.2, color='lightblue')
    
    # Add some quantum states
    states = [
        ([0, 0, 1], '|0⟩'),
        ([0, 0, -1], '|1⟩'),
        ([1, 0, 0], '|+⟩'),
        ([0, 1, 0], '|i⟩')
    ]
    
    for (x, y, z), label in states:
        ax3.scatter([x], [y], [z], s=100, c='red')
        ax3.text(x*1.1, y*1.1, z*1.1, label, fontsize=12)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('Quantum State Space (Bloch Sphere)')
    
    # 4. Hybrid algorithm workflow
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.9, 'Quantum-Classical Hybrid Workflow', 
             ha='center', va='center', fontsize=14, weight='bold')
    
    workflow_steps = [
        'Classical preprocessing',
        'Quantum circuit generation',
        'QPU execution',
        'Classical postprocessing',
        'Result optimization'
    ]
    
    y_pos = 0.7
    for i, step in enumerate(workflow_steps):
        color = colors[3] if 'Quantum' in step or 'QPU' in step else colors[0]
        ax4.text(0.1, y_pos, f"{i+1}. {step}", fontsize=12, color=color)
        y_pos -= 0.15
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demo function"""
    print("FPGA Quantum Algorithm Examples")
    print("=" * 50)
    
    # Check quantum availability
    if not QUANTUM_AVAILABLE:
        print("\nWARNING: Quantum features not available.")
        print("Install qiskit for full functionality: pip install qiskit")
        print("Running in classical simulation mode.\n")
    
    # Create FPGA with quantum capabilities
    print("Initializing FPGA with Quantum Processing Unit...")
    fpga = AdvancedFPGAFabric(rows=8, cols=8, num_brams=4, num_dsps=4)
    
    # Create demo instance
    demo = QuantumSearchDemo(fpga)
    
    # Run demonstrations
    try:
        demo.demo_grover_search()
        demo.demo_quantum_fourier_transform()
        demo.demo_variational_quantum_eigensolver()
        demo.demo_quantum_phase_estimation()
        demo.demo_quantum_machine_learning()
        
        # Benchmarks
        benchmark_quantum_algorithms(fpga)
        
        # Visualization
        visualize_quantum_fpga_integration(fpga)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("Some features may require additional setup or dependencies.")
    
    print("\nQuantum algorithm demonstrations complete!")
    print("The FPGA's QPU accelerates quantum algorithms while classical")
    print("components handle pre/post-processing and hybrid algorithms.")


if __name__ == "__main__":
    main()
