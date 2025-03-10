#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
import simple_af_ofdm  # Compiled ArrayFire OFDM module
import simple_tf_ofdm  # TensorFlow OFDM module
import tensorflow as tf

def af_to_tf(array):
    """
    Converts an ArrayFire output (in Fortran/column-major order) into TensorFlow's row-major order.
    """
    flat = np.ravel(array, order='C')
    return np.reshape(flat, array.shape, order='F')

def benchmark_tf_ofdm(inputs_list, fft_size, cp, l_min=-2):
    """
    Runs the TensorFlow OFDM demodulator on the provided list of inputs.
    Returns the average runtime and a list of output grids.
    """
    times = []
    outputs = []
    # Warm-up run
    _ = simple_tf_ofdm.tf_ofdm_demodulator(inputs_list[0], fft_size, l_min, cp)
    for inp in inputs_list:
        start = time.perf_counter()
        grid = simple_tf_ofdm.tf_ofdm_demodulator(inp, fft_size, l_min, cp)
        end = time.perf_counter()
        times.append(end - start)
        grid = grid.numpy() if isinstance(grid, tf.Tensor) else grid
        outputs.append(grid)
    avg_time = sum(times) / len(times)
    return avg_time, outputs

def benchmark_af_ofdm(inputs_list, fft_size, cp, l_min=-2):
    """
    Runs the ArrayFire OFDM demodulator on the provided list of inputs.
    Returns the average runtime and a list of output grids.
    """
    times = []
    outputs = []
    # Warm-up run (convert the first input to Fortran order)
    inp_for = np.asfortranarray(inputs_list[0])
    _ = simple_af_ofdm.af_ofdm_demodulator(inp_for, fft_size, l_min, cp)
    for inp in inputs_list:
        inp_for = np.asfortranarray(inp)
        start = time.perf_counter()
        grid = simple_af_ofdm.af_ofdm_demodulator(inp_for, fft_size, l_min, cp)
        end = time.perf_counter()
        times.append(end - start)
        outputs.append(grid)
    avg_time = sum(times) / len(times)
    return avg_time, outputs

def main():
    # Define FFT sizes to test.
    fft_sizes = [16, 32, 64, 128, 256, 512, 1024]
    num_runs = 5       # Number of runs per FFT size
    num_symbols = 30   # Number of OFDM symbols per run
    l_min = -2         # Largest negative time lag
    speedups = []

    for fft_size in fft_sizes:
        # We set CP to 1/4 of fft_size (but at least 1 sample)
        cp = max(1, fft_size // 4)
        symbol_length = fft_size + cp
        total_length = num_symbols * symbol_length

        print(f"\n=== FFT Size = {fft_size}, CP = {cp}, Symbols = {num_symbols} ===")
        # Generate a common list of random inputs (1D complex signals) for this FFT size.
        inputs_list = [
            (np.random.rand(total_length) + 1j * np.random.rand(total_length)).astype(np.complex64)
            for _ in range(num_runs)
        ]

        # Benchmark TensorFlow using the common inputs.
        tf_time, tf_outputs = benchmark_tf_ofdm(inputs_list, fft_size, cp, l_min)
        # Benchmark ArrayFire using the same common inputs.
        af_time, af_outputs = benchmark_af_ofdm(inputs_list, fft_size, cp, l_min)

        print(f"TensorFlow average time: {tf_time:.6f} seconds")
        print(f"ArrayFire average time:    {af_time:.6f} seconds")
        speedup = tf_time / max(af_time, 1e-9)
        print(f"AF over TF Speedup:        {speedup:.3f}x")
        speedups.append(speedup)

        # Convert AF output into TF's row-major ordering.
        af_outputs = [af_to_tf(res) for res in af_outputs]

        # Compare results from the first run.
        tf_result = tf_outputs[0]
        af_result = af_outputs[0]

        print("Sample 3x3 submatrix (TF):")
        print(tf_result[:3, :3])
        print("Sample 3x3 submatrix (AF):")
        print(af_result[:3, :3])
        if np.allclose(tf_result, af_result, rtol=1e-3, atol=1e-3):
            print("Results match within tolerance!")
        else:
            print("Results differ numerically!")

    # Plot speedup versus FFT size.
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 8))
    plt.plot(fft_sizes, speedups, marker='o', linestyle='-', color='b')
    plt.xlabel("OFDM Subcarrier Count")
    plt.ylabel("AF over TF Speedup")
    plt.title("ArrayFire vs. TensorFlow OFDM Demodulator - RTX 2000 Ada")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
