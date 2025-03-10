#!/usr/bin/env python3
import time
import numpy as np
import simple_af  # the compiled ArrayFire combined module
import simple_tf  # the TensorFlow combined module
import matplotlib.pyplot as plt

def benchmark_af_list(y_list, h_list, s_list):
    times = []
    results = []
    total_list = list(zip(y_list, h_list, s_list))

    # Warm-up using the first input pair (if available)
    if len(y_list) > 0:
        y, h, s = total_list[0]
        y_for = np.asfortranarray(y)
        h_for = np.asfortranarray(h)
        s_for = np.asfortranarray(s)
        x_hat_af, no_eff_af = simple_af.af_lmmse_equalizer(y_for, h_for, s_for)

    for (y, h, s) in total_list:
        y_for = np.asfortranarray(y)
        h_for = np.asfortranarray(h)
        s_for = np.asfortranarray(s)

        start = time.perf_counter()
        x_hat_af, no_eff_af = simple_af.af_lmmse_equalizer(y_for, h_for, s_for)
        # Reshape the result to match TensorFlow's output dimensions
        x_hat_af = x_hat_af.reshape(x_hat_af.shape[::-1]).T
        end = time.perf_counter()

        times.append(end - start)
        results.append((x_hat_af, no_eff_af))

    avg_time = sum(times) / len(times)
    return avg_time, results

def benchmark_tf_list(y_list, h_list, s_list):
    times = []
    results = []
    total_list = list(zip(y_list, h_list, s_list))

    # Warm-up using the first input pair (if available)
    if len(y_list) > 0:
        y, h, s = total_list[0]
        y_for = np.asfortranarray(y)
        h_for = np.asfortranarray(h)
        s_for = np.asfortranarray(s)
        x_hat_tf, no_eff_tf = simple_tf.tf_lmmse_equalizer(y_for, h_for, s_for)
    
    for (y, h, s) in total_list:
        y_for = np.asfortranarray(y)
        h_for = np.asfortranarray(h)
        s_for = np.asfortranarray(s)

        start = time.perf_counter()
        x_hat_tf, no_eff_tf = simple_tf.tf_lmmse_equalizer(y_for, h_for, s_for)
        end = time.perf_counter()

        times.append(end - start)
        results.append((x_hat_tf, no_eff_tf))

    avg_time = sum(times) / len(times)
    return avg_time, results

def main():
    # List of matrix sizes to test (transmit antenna counts)
    Ks = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 192, 256, 512, 768, 1024]
    num_runs = 5  # Number of runs (new random inputs) per size
    relative_differences = []  # To store average relative difference for each K

    for K in Ks:
        M = 4 * K  # Twice as many receive antennas as transmit antennas
        T = 20

        y_list = []
        h_list = []
        s_list = []

        for _ in range(num_runs):
            # Create random complex inputs.
            y = (np.random.rand(M, T) + 1j * np.random.rand(M, T)).astype(np.complex64)
            h = (np.random.rand(M, K) + 1j * np.random.rand(M, K)).astype(np.complex64)
            # For s, use a scaled identity (ensuring positive-definiteness).
            s = (np.eye(M, dtype=np.complex64)) * 0.1
            y_list.append(y)
            h_list.append(h)
            s_list.append(s)

        print("=== TensorFlow LMMSE Equalizer ===")
        tf_time, tf_results = benchmark_tf_list(y_list, h_list, s_list)
        print("=== ArrayFire LMMSE Equalizer ===")
        af_time, af_results = benchmark_af_list(y_list, h_list, s_list)

        run_rel_diffs = []
        all_match = True
        for i, (r_tf, r_af) in enumerate(zip(tf_results, af_results)):
            x_tf, no_tf = r_tf
            x_af, no_af = r_af
            # Compute the relative difference using the Frobenius norm.
            rel_diff = np.linalg.norm(x_tf - x_af) / np.linalg.norm(x_tf)
            run_rel_diffs.append(rel_diff)
            # Check the closeness of the results for diagnostic purposes.
            if not np.allclose(x_af, x_tf, rtol=1e-3, atol=1e-3):
                if not np.allclose(x_af, x_tf, rtol=1e-1, atol=1e-1):
                    print(f"Run {i}: The combined results differ significantly!")
                    all_match = False
                else:
                    print(f"Run {i}: The combined results should be inspected more closely.")
        avg_rel_diff = np.mean(run_rel_diffs)
        relative_differences.append(avg_rel_diff)

        if all_match:
            print(f"All combined results match within tolerance for {K} transmit antennas!")
        else:
            print(f"Some combined results differ for {K} transmit antennas; please inspect further.")

        print("ArrayFire result (3x3 submatrix) from run 0:")
        print(af_results[0][0][:3, :3])
        print("TensorFlow result (3x3 submatrix) from run 0:")
        print(tf_results[0][0][:3, :3])
        print(f"Average relative difference for K = {K}: {avg_rel_diff:.6f}\n")

    # Plot the average relative difference vs. the transmit antenna count.
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 8))
    plt.plot(Ks, relative_differences, marker='o', linestyle='-', color='b')
    plt.xlabel("Transmit Antenna Count")
    plt.ylabel("Relative Difference")
    plt.title("Relative Difference between AF and TF LMMSE Equalizer")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
