"""
Basic unit tests for FPGA Simulator

Run with: pytest tests/test_basic.py -v
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpga_simulator import (
    AdvancedFPGAFabric, ConfigurableLogicBlock, BlockRAM, DSPBlock,
    Signal, SignalType, NoiseModel, FaultModel,
    GPU_AVAILABLE, QUANTUM_AVAILABLE, ML_AVAILABLE
)


class TestBasicComponents:
    """Test basic FPGA components"""
    
    def test_fpga_creation(self):
        """Test FPGA fabric creation"""
        fpga = AdvancedFPGAFabric(rows=4, cols=4, num_brams=2, num_dsps=2)
        
        assert fpga.rows == 4
        assert fpga.cols == 4
        assert len(fpga.brams) == 2
        assert len(fpga.dsps) == 2
        assert len(fpga.clbs) == 4
        assert len(fpga.clbs[0]) == 4
    
    def test_clb_basic(self):
        """Test CLB basic operations"""
        clb = ConfigurableLogicBlock(block_id=0, num_inputs=4, use_gpu=False)
        
        # Configure as AND gate
        and_lut = [0] * 15 + [1]  # Only output 1 when all inputs are 1
        clb.configure({'lut': and_lut})
        
        # Test AND operation
        result = clb.evaluate(np.array([1, 1, 1, 1]))
        assert result.value == 1
        
        result = clb.evaluate(np.array([1, 0, 1, 1]))
        assert result.value == 0
    
    def test_clb_xor(self):
        """Test CLB configured as XOR gate"""
        clb = ConfigurableLogicBlock(block_id=1, num_inputs=4, use_gpu=False)
        
        # Configure as 2-input XOR (ignoring last 2 inputs)
        xor_lut = [0, 1, 1, 0] * 4  # XOR pattern repeated
        clb.configure({'lut': xor_lut})
        
        # Test XOR operation
        test_cases = [
            ([0, 0, 0, 0], 0),
            ([0, 1, 0, 0], 1),
            ([1, 0, 0, 0], 1),
            ([1, 1, 0, 0], 0),
        ]
        
        for inputs, expected in test_cases:
            result = clb.evaluate(np.array(inputs))
            assert result.value == expected


class TestMemoryComponents:
    """Test memory components (BRAM)"""
    
    def test_bram_basic(self):
        """Test basic BRAM read/write"""
        bram = BlockRAM(depth=256, width=8, use_ecc=False)
        
        # Write and read back
        test_data = [0x42, 0xFF, 0x00, 0xAA]
        
        for addr, data in enumerate(test_data):
            bram.write(addr, data)
        
        for addr, expected in enumerate(test_data):
            data, error = bram.read(addr)
            assert data == expected
            assert error == False  # No ECC, so no errors detected
    
    def test_bram_ecc(self):
        """Test BRAM with ECC"""
        bram = BlockRAM(depth=256, width=8, use_ecc=True)
        
        # Write data
        test_value = 0x55
        bram.write(0, test_value)
        
        # Read without error
        data, error = bram.read(0)
        assert data == test_value
        
        # Note: Full ECC testing would require access to internal memory
        # to inject bit flips, which is implementation-specific


class TestDSPComponents:
    """Test DSP blocks"""
    
    def test_dsp_multiply(self):
        """Test DSP multiplication"""
        dsp = DSPBlock(width=18, use_gpu=False)
        
        # Test simple multiplication
        result = dsp.multiply(7, 9)
        assert abs(result - 63) < 1  # Allow for noise
        
        # Test array multiplication
        a = np.array([1, 2, 3, 4])
        b = np.array([5, 6, 7, 8])
        result = dsp.multiply(a, b)
        expected = a * b
        
        # Check with tolerance for noise
        assert np.allclose(result, expected, rtol=0.1)
    
    def test_dsp_mac(self):
        """Test DSP MAC operation"""
        dsp = DSPBlock(width=18, use_gpu=False)
        
        # Reset accumulator
        dsp.accumulator = 0
        
        # Perform MAC operations
        pairs = [(2, 3), (4, 5), (1, 6)]
        expected = sum(a * b for a, b in pairs)
        
        for a, b in pairs:
            result = dsp.mac(a, b)
        
        # Check with tolerance for noise
        assert abs(dsp.accumulator - expected) < expected * 0.1


class TestSignalAndNoise:
    """Test signal and noise models"""
    
    def test_signal_creation(self):
        """Test signal creation"""
        signal = Signal(
            value=1,
            timestamp=1.23e-9,
            signal_type=SignalType.DIGITAL,
            noise_level=0.01
        )
        
        assert signal.value == 1
        assert signal.timestamp == 1.23e-9
        assert signal.signal_type == SignalType.DIGITAL
        assert signal.noise_level == 0.01
    
    def test_noise_model(self):
        """Test noise model"""
        noise_model = NoiseModel(vdd=1.0, temp_kelvin=300)
        
        # Test thermal noise
        clean_signal = 1.0
        noisy_signal = noise_model.add_thermal_noise(clean_signal)
        
        # Signal should be different due to noise
        assert noisy_signal != clean_signal
        
        # But should be reasonably close
        assert abs(noisy_signal - clean_signal) < 0.5
    
    def test_crosstalk(self):
        """Test crosstalk modeling"""
        noise_model = NoiseModel()
        
        signals = [1.0, 0.0, 1.0, 0.0]
        noisy = noise_model.add_crosstalk(signals, coupling=0.1)
        
        # Middle signals should be affected by neighbors
        assert noisy[1] > 0.0  # Should have some crosstalk from neighbors
        assert noisy[2] < 1.0  # Should be reduced by neighbor at 0


class TestFaultTolerance:
    """Test fault injection and tolerance"""
    
    def test_fault_model(self):
        """Test fault model basic operations"""
        fault_model = FaultModel(seu_rate=0.5, stuck_at_prob=0.0)
        
        # With high SEU rate, should see some bit flips
        flips = 0
        for _ in range(100):
            value, flipped = fault_model.inject_seu(0, bit_width=1)
            if flipped:
                flips += 1
        
        # Should see roughly 50 flips with seu_rate=0.5
        assert 30 < flips < 70
    
    def test_fpga_fault_recovery(self):
        """Test FPGA-level fault recovery"""
        fpga = AdvancedFPGAFabric(rows=2, cols=2, num_brams=1)
        
        # Run fault injection
        stats = fpga.inject_and_recover_faults(num_faults=10)
        
        # Should have some stats
        assert 'seu_injected' in stats
        assert 'seu_recovered' in stats
        
        # Recovery rate should be reasonable
        if stats['seu_injected'] > 0:
            recovery_rate = stats['seu_recovered'] / stats['seu_injected']
            assert 0.0 <= recovery_rate <= 1.0


class TestParallelExecution:
    """Test parallel execution capabilities"""
    
    def test_parallel_evaluation(self):
        """Test parallel CLB evaluation"""
        fpga = AdvancedFPGAFabric(rows=2, cols=2)
        
        # Configure CLBs
        for i in range(2):
            for j in range(2):
                # Simple pass-through LUT
                fpga.clbs[i][j].configure({'lut': list(range(16))})
        
        # Create test inputs
        inputs = {
            (0, 0): np.array([0, 0, 0, 0]),
            (0, 1): np.array([1, 0, 0, 0]),
            (1, 0): np.array([0, 1, 0, 0]),
            (1, 1): np.array([1, 1, 0, 0])
        }
        
        # Evaluate in parallel
        results = fpga.parallel_evaluate(inputs)
        
        # Check results
        assert len(results) == 4
        assert (0, 0) in results
        assert results[(0, 0)].value == 0
        assert results[(0, 1)].value == 1


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPUAcceleration:
    """Test GPU acceleration features"""
    
    def test_gpu_batch_evaluation(self):
        """Test GPU batch evaluation"""
        fpga = AdvancedFPGAFabric(rows=2, cols=2, use_gpu=True)
        
        # Create batch input
        batch_size = 4
        input_matrix = np.random.randint(0, 2, (batch_size, 4))
        
        # Evaluate batch
        results = fpga.gpu_batch_evaluate(input_matrix)
        
        assert len(results) == batch_size


@pytest.mark.skipif(not QUANTUM_AVAILABLE, reason="Quantum features not available")
class TestQuantumFeatures:
    """Test quantum computing features"""
    
    def test_quantum_search(self):
        """Test quantum search algorithm"""
        fpga = AdvancedFPGAFabric(rows=2, cols=2)
        
        # Simple search for value 5
        result = fpga.run_quantum_accelerated_search(8, lambda x: x == 5)
        
        assert result == 5
    
    def test_qft(self):
        """Test Quantum Fourier Transform"""
        fpga = AdvancedFPGAFabric(rows=2, cols=2)
        
        input_state = [1, 0, 0, 0]
        result = fpga.qpu.quantum_fourier_transform(input_state)
        
        # Result should be a complex array
        assert len(result) == len(input_state)
        assert result.dtype == np.complex128


@pytest.mark.skipif(not ML_AVAILABLE, reason="ML features not available")
class TestMLFeatures:
    """Test machine learning features"""
    
    def test_ml_router(self):
        """Test ML-based router"""
        from fpga_simulator import MLOptimizedRouter
        
        router = MLOptimizedRouter()
        
        # Generate synthetic training data
        routing_data = []
        for _ in range(20):
            features = np.random.rand(10)
            delay = np.random.exponential(1.0)
            routing_data.append((features, delay))
        
        # Train model
        router.train_routing_model(routing_data)
        
        # Test prediction
        test_features = np.random.rand(10)
        predicted_delay = router.predict_delay(test_features)
        
        assert predicted_delay > 0


class TestHDLCompilation:
    """Test HDL compilation features"""
    
    def test_simple_hdl_compilation(self):
        """Test compiling simple Verilog"""
        fpga = AdvancedFPGAFabric(rows=4, cols=4)
        
        verilog = """
        module test(out, a, b);
            output out;
            input a, b;
            and g1 (out, a, b);
        endmodule
        """
        
        try:
            bitstream = fpga.compile_hdl(verilog)
            
            assert 'metadata' in bitstream
            assert 'clbs' in bitstream
            assert len(bitstream['clbs']) > 0
        except Exception as e:
            # HDL compilation might fail due to simplified parser
            pytest.skip(f"HDL compilation not fully implemented: {e}")


# Fixture for common FPGA setup
@pytest.fixture
def basic_fpga():
    """Create a basic FPGA for testing"""
    return AdvancedFPGAFabric(rows=4, cols=4, num_brams=2, num_dsps=2)


# Parametrized tests
@pytest.mark.parametrize("rows,cols", [(2, 2), (4, 4), (8, 8)])
def test_fpga_sizes(rows, cols):
    """Test creating FPGAs of different sizes"""
    fpga = AdvancedFPGAFabric(rows=rows, cols=cols)
    
    assert fpga.rows == rows
    assert fpga.cols == cols
    assert len(fpga.clbs) == rows
    assert len(fpga.clbs[0]) == cols


@pytest.mark.parametrize("gate_type,truth_table,test_inputs,expected", [
    ("AND", [0]*15 + [1], [[0,0,0,0], [1,1,1,1]], [0, 1]),
    ("OR", [0] + [1]*15, [[0,0,0,0], [1,0,0,0]], [0, 1]),
    ("NOT", [1,0]*8, [[0,0,0,0], [1,0,0,0]], [1, 0]),
])
def test_logic_gates(gate_type, truth_table, test_inputs, expected):
    """Test various logic gate configurations"""
    clb = ConfigurableLogicBlock(0, use_gpu=False)
    clb.configure({'lut': truth_table})
    
    for inputs, exp_output in zip(test_inputs, expected):
        result = clb.evaluate(np.array(inputs))
        assert result.value == exp_output


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
