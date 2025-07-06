#!/usr/bin/env python3
"""
Neural Network Accelerator Example for FPGA Simulator

This example demonstrates how to use the FPGA simulator to accelerate
neural network inference by implementing common NN operations on FPGA.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time
import sys
import os

# Add parent directory to path to import fpga_simulator
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fpga_simulator import AdvancedFPGAFabric, Signal, SignalType
except ImportError:
    print("Error: Could not import fpga_simulator. Make sure it's in the parent directory.")
    sys.exit(1)


class FPGANeuralNetwork:
    """Neural Network implemented on FPGA fabric"""
    
    def __init__(self, fpga: AdvancedFPGAFabric, input_size: int, 
                 hidden_size: int, output_size: int):
        self.fpga = fpga
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Allocate FPGA resources
        self._allocate_resources()
        
        # Initialize weights (quantized to 8-bit)
        self._initialize_weights()
        
    def _allocate_resources(self):
        """Allocate CLBs and DSPs for NN operations"""
        # Use CLBs for activation functions
        self.activation_clbs = []
        for i in range(self.hidden_size + self.output_size):
            row = i // self.fpga.cols
            col = i % self.fpga.cols
            if row < self.fpga.rows:
                self.activation_clbs.append((row, col))
        
        # Use DSPs for matrix multiplication
        self.available_dsps = list(range(len(self.fpga.dsps)))
        
    def _initialize_weights(self):
        """Initialize quantized weights"""
        # Random initialization scaled for 8-bit representation
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * 0.5
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * 0.5
        
        # Quantize to 8-bit
        self.w1_quantized = self._quantize_weights(self.w1)
        self.w2_quantized = self._quantize_weights(self.w2)
        
        # Store in BRAM if available
        if len(self.fpga.brams) > 0:
            self._store_weights_in_bram()
    
    def _quantize_weights(self, weights: np.ndarray, bits: int = 8) -> np.ndarray:
        """Quantize floating point weights to fixed point"""
        scale = (2**(bits-1) - 1) / np.max(np.abs(weights))
        quantized = np.round(weights * scale).astype(np.int8)
        return quantized
    
    def _store_weights_in_bram(self):
        """Store weights in BRAM for fast access"""
        print("Storing weights in BRAM...")
        
        # Flatten weights and store in BRAM
        w1_flat = self.w1_quantized.flatten()
        w2_flat = self.w2_quantized.flatten()
        
        # Store in first BRAM
        bram = self.fpga.brams[0]
        addr = 0
        
        # Store first layer weights
        for weight in w1_flat:
            if addr < bram.depth:
                bram.write(addr, int(weight) & 0xFF)
                addr += 1
        
        self.w1_bram_start = 0
        self.w1_bram_end = addr
        
        # Store second layer weights
        for weight in w2_flat:
            if addr < bram.depth:
                bram.write(addr, int(weight) & 0xFF)
                addr += 1
                
        self.w2_bram_start = self.w1_bram_end
        self.w2_bram_end = addr
        
        print(f"Stored {addr} weight values in BRAM")
    
    def _configure_activation_lut(self, clb_pos: Tuple[int, int], 
                                 activation: str = 'relu'):
        """Configure CLB LUT for activation function"""
        row, col = clb_pos
        clb = self.fpga.clbs[row][col]
        
        if activation == 'relu':
            # ReLU: max(0, x)
            # For 4-bit input, threshold at 8 (middle)
            relu_lut = []
            for i in range(16):
                if i < 8:
                    relu_lut.append(0)  # Negative -> 0
                else:
                    relu_lut.append(i - 8)  # Positive -> scaled value
                    
            clb.configure({'lut': relu_lut})
            
        elif activation == 'sigmoid':
            # Approximate sigmoid with LUT
            sigmoid_lut = []
            for i in range(16):
                x = (i - 8) / 4.0  # Scale to [-2, 2]
                y = 1 / (1 + np.exp(-x))
                sigmoid_lut.append(int(y * 15))  # Scale to [0, 15]
                
            clb.configure({'lut': sigmoid_lut})
            
        elif activation == 'tanh':
            # Approximate tanh with LUT
            tanh_lut = []
            for i in range(16):
                x = (i - 8) / 4.0  # Scale to [-2, 2]
                y = np.tanh(x)
                tanh_lut.append(int((y + 1) * 7.5))  # Scale to [0, 15]
                
            clb.configure({'lut': tanh_lut})
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        start_time = time.time()
        
        # Layer 1: Input -> Hidden
        hidden = self._matrix_multiply_dsp(x, self.w1_quantized)
        hidden = self._apply_activation(hidden, 'relu')
        
        # Layer 2: Hidden -> Output
        output = self._matrix_multiply_dsp(hidden, self.w2_quantized)
        output = self._apply_activation(output, 'sigmoid')
        
        inference_time = time.time() - start_time
        print(f"Inference time: {inference_time*1000:.2f} ms")
        
        return output
    
    def _matrix_multiply_dsp(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Matrix multiplication using DSP blocks"""
        result = np.zeros(w.shape[1])
        
        # Distribute computation across DSP blocks
        if len(self.fpga.dsps) > 0:
            outputs_per_dsp = w.shape[1] // len(self.fpga.dsps)
            
            for dsp_idx, dsp in enumerate(self.fpga.dsps):
                start_col = dsp_idx * outputs_per_dsp
                end_col = start_col + outputs_per_dsp
                if dsp_idx == len(self.fpga.dsps) - 1:
                    end_col = w.shape[1]
                
                # Compute partial results
                for j in range(start_col, end_col):
                    # Reset accumulator
                    dsp.accumulator = 0
                    
                    # MAC operations
                    for i in range(len(x)):
                        dsp.mac(x[i], w[i, j])
                    
                    result[j] = dsp.accumulator
        else:
            # Fallback to numpy
            result = x @ w
            
        return result
    
    def _apply_activation(self, x: np.ndarray, activation: str) -> np.ndarray:
        """Apply activation function using configured CLBs"""
        activated = np.zeros_like(x)
        
        for i, value in enumerate(x):
            if i < len(self.activation_clbs):
                row, col = self.activation_clbs[i]
                clb = self.fpga.clbs[row][col]
                
                # Configure activation if not already done
                self._configure_activation_lut((row, col), activation)
                
                # Quantize input to 4 bits for LUT
                quantized_input = int(np.clip(value * 2 + 8, 0, 15))
                
                # Evaluate through CLB
                signal = clb.evaluate(np.array([
                    quantized_input & 1,
                    (quantized_input >> 1) & 1,
                    (quantized_input >> 2) & 1,
                    (quantized_input >> 3) & 1
                ]))
                
                activated[i] = signal.value / 15.0  # Normalize back
            else:
                # Fallback for remaining values
                if activation == 'relu':
                    activated[i] = max(0, value)
                elif activation == 'sigmoid':
                    activated[i] = 1 / (1 + np.exp(-value))
                elif activation == 'tanh':
                    activated[i] = np.tanh(value)
                    
        return activated
    
    def train_on_fpga(self, X: np.ndarray, y: np.ndarray, 
                      epochs: int = 100, lr: float = 0.01):
        """Simple training using FPGA acceleration"""
        print(f"Training on FPGA for {epochs} epochs...")
        
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss (MSE)
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            # Backward pass (simplified - using numpy for gradients)
            # In real implementation, this would also use FPGA
            output_error = output - y
            
            # Update weights (simplified)
            # This is where you'd implement backprop on FPGA
            self.w2 -= lr * np.outer(self._hidden_state, output_error)
            self.w1 -= lr * np.outer(X, self._hidden_error)
            
            # Re-quantize weights
            self.w1_quantized = self._quantize_weights(self.w1)
            self.w2_quantized = self._quantize_weights(self.w2)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses


def demo_xor_network():
    """Demo: Train a network to learn XOR function"""
    print("=== XOR Learning Demo ===")
    
    # Create FPGA
    fpga = AdvancedFPGAFabric(rows=8, cols=8, num_brams=2, num_dsps=4)
    
    # Create neural network
    nn = FPGANeuralNetwork(fpga, input_size=2, hidden_size=4, output_size=1)
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Test inference
    print("\nBefore training:")
    for i in range(len(X)):
        output = nn.forward(X[i])
        print(f"Input: {X[i]} -> Output: {output[0]:.3f} (Expected: {y[i][0]})")
    
    # Visualize network architecture
    visualize_nn_on_fpga(fpga, nn)
    
    return fpga, nn


def demo_pattern_recognition():
    """Demo: Simple pattern recognition"""
    print("\n=== Pattern Recognition Demo ===")
    
    # Create FPGA
    fpga = AdvancedFPGAFabric(rows=8, cols=8, num_brams=4, num_dsps=4)
    
    # Create larger network
    nn = FPGANeuralNetwork(fpga, input_size=16, hidden_size=8, output_size=4)
    
    # Create simple patterns (4x4 images)
    patterns = {
        'horizontal': np.array([
            1, 1, 1, 1,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0
        ]),
        'vertical': np.array([
            1, 0, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 0,
            1, 0, 0, 0
        ]),
        'diagonal': np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]),
        'cross': np.array([
            1, 0, 0, 1,
            0, 1, 1, 0,
            0, 1, 1, 0,
            1, 0, 0, 1
        ])
    }
    
    # Test pattern recognition
    print("\nPattern recognition results:")
    for name, pattern in patterns.items():
        output = nn.forward(pattern)
        predicted_class = np.argmax(output)
        confidence = output[predicted_class]
        print(f"{name}: Class {predicted_class} (confidence: {confidence:.2f})")
        
        # Visualize pattern
        plt.figure(figsize=(3, 3))
        plt.imshow(pattern.reshape(4, 4), cmap='binary')
        plt.title(f"{name} -> Class {predicted_class}")
        plt.axis('off')
        plt.show()
    
    return fpga, nn


def benchmark_fpga_vs_cpu():
    """Benchmark FPGA vs CPU performance"""
    print("\n=== Performance Benchmark ===")
    
    # Test different network sizes
    sizes = [(10, 20, 10), (50, 100, 50), (100, 200, 100)]
    
    for input_size, hidden_size, output_size in sizes:
        print(f"\nNetwork size: {input_size} -> {hidden_size} -> {output_size}")
        
        # Create FPGA network
        fpga = AdvancedFPGAFabric(rows=16, cols=16, num_brams=8, num_dsps=8)
        nn_fpga = FPGANeuralNetwork(fpga, input_size, hidden_size, output_size)
        
        # Create test data
        test_data = np.random.randn(100, input_size)
        
        # Benchmark FPGA
        start = time.time()
        for data in test_data:
            _ = nn_fpga.forward(data)
        fpga_time = time.time() - start
        
        # Benchmark CPU (numpy)
        w1 = np.random.randn(input_size, hidden_size)
        w2 = np.random.randn(hidden_size, output_size)
        
        start = time.time()
        for data in test_data:
            hidden = np.maximum(0, data @ w1)  # ReLU
            output = 1 / (1 + np.exp(-hidden @ w2))  # Sigmoid
        cpu_time = time.time() - start
        
        print(f"FPGA time: {fpga_time*1000:.2f} ms")
        print(f"CPU time: {cpu_time*1000:.2f} ms")
        print(f"Speedup: {cpu_time/fpga_time:.2f}x")
        
        # Show resource utilization
        metrics = fpga.get_performance_metrics()
        print(f"FPGA utilization: {metrics['total_evaluations']} evaluations")


def visualize_nn_on_fpga(fpga: AdvancedFPGAFabric, nn: FPGANeuralNetwork):
    """Visualize how the neural network is mapped to FPGA resources"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Visualize CLB allocation for activations
    clb_grid = np.zeros((fpga.rows, fpga.cols))
    for row, col in nn.activation_clbs:
        clb_grid[row, col] = 1
    
    im1 = ax1.imshow(clb_grid, cmap='YlOrRd', vmin=0, vmax=1)
    ax1.set_title('CLB Allocation for Activation Functions')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # Add grid
    for i in range(fpga.rows + 1):
        ax1.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(fpga.cols + 1):
        ax1.axvline(j - 0.5, color='gray', linewidth=0.5)
    
    # Visualize DSP usage
    dsp_usage = ['DSP {}: Matrix Multiply'.format(i) for i in range(len(fpga.dsps))]
    bram_usage = ['BRAM {}: Weight Storage'.format(i) for i in range(min(2, len(fpga.brams)))]
    
    ax2.text(0.1, 0.9, 'Resource Allocation:', fontsize=14, weight='bold', 
             transform=ax2.transAxes)
    
    y_pos = 0.8
    for usage in dsp_usage:
        ax2.text(0.1, y_pos, usage, fontsize=12, transform=ax2.transAxes)
        y_pos -= 0.1
        
    y_pos -= 0.05
    for usage in bram_usage:
        ax2.text(0.1, y_pos, usage, fontsize=12, transform=ax2.transAxes)
        y_pos -= 0.1
    
    # Add network architecture
    y_pos -= 0.1
    ax2.text(0.1, y_pos, f'Network Architecture:', fontsize=14, weight='bold',
             transform=ax2.transAxes)
    ax2.text(0.1, y_pos-0.1, f'Input: {nn.input_size} neurons', fontsize=12,
             transform=ax2.transAxes)
    ax2.text(0.1, y_pos-0.2, f'Hidden: {nn.hidden_size} neurons', fontsize=12,
             transform=ax2.transAxes)
    ax2.text(0.1, y_pos-0.3, f'Output: {nn.output_size} neurons', fontsize=12,
             transform=ax2.transAxes)
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demo function"""
    print("FPGA Neural Network Accelerator Demo")
    print("=" * 50)
    
    # Run demos
    demo_xor_network()
    demo_pattern_recognition()
    benchmark_fpga_vs_cpu()
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
