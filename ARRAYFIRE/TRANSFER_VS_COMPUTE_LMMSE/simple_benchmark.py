#!/usr/bin/env python3
import time
import numpy as np
import simple_af  # your split module exposing upload_data, compute_equalizer, download_data
import matplotlib.pyplot as plt

def benchmark_transfer_vs_compute(y_list, h_list, s_list):
    upload_times = []
    compute_times = []
    download_times = []

    # Warm-up: run one iteration to “prime” the GPU and caches.
    if len(y_list) > 0:
        simple_af.upload_data(y_list[0], h_list[0], s_list[0])
        simple_af.compute_equalizer()
        simple_af.download_data()

    # Timed runs.
    for (y, h, s) in zip(y_list, h_list, s_list):
        t0 = time.perf_counter()
        simple_af.upload_data(y, h, s)
        t1 = time.perf_counter()
        simple_af.compute_equalizer()
        t2 = time.perf_counter()
        x_hat, no = simple_af.download_data()
        t3 = time.perf_counter()

        upload_times.append(t1 - t0)
        compute_times.append(t2 - t1)
        download_times.append(t3 - t2)

    avg_upload = sum(upload_times) / len(upload_times)
    avg_compute = sum(compute_times) / len(compute_times)
    avg_download = sum(download_times) / len(download_times)
    total_transfer = avg_upload + avg_download

    return total_transfer, avg_compute

def main():
    # Define a set of matrix sizes (transmit antenna counts)
    Ks = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 192, 256, 512, 768, 1024]
    num_runs = 5  # Number of runs per size
    ratio_list = []  # Will store the (transfer time / compute time) ratio for each K

    for K in Ks:
        M = 2 * K  # Twice as many receive antennas as transmit antennas
        T = 20     # Number of symbols per run

        y_list = []
        h_list = []
        s_list = []
        for _ in range(num_runs):
            # Generate random complex inputs.
            y = (np.random.rand(M, T) + 1j * np.random.rand(M, T)).astype(np.complex64)
            h = (np.random.rand(M, K) + 1j * np.random.rand(M, K)).astype(np.complex64)
            # Use a scaled identity for the noise covariance (ensuring positive-definiteness)
            s = (np.eye(M, dtype=np.complex64)) * 0.1
            y_list.append(y)
            h_list.append(h)
            s_list.append(s)

        total_transfer, avg_compute = benchmark_transfer_vs_compute(y_list, h_list, s_list)
        ratio = total_transfer / avg_compute
        print(f"K = {K:4d}: Transfer (upload+download) = {total_transfer:.6f} s, "
              f"Compute = {avg_compute:.6f} s, Ratio = {ratio:.6f}")
        ratio_list.append(ratio)

    # Plot the relative ratio (transfer time / compute time) vs. transmit antenna count.
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 8))
    plt.plot(Ks, ratio_list, marker='o', linestyle='-', color='b')
    plt.xlabel("Transmit Antenna Count (K)")
    plt.ylabel("Transfer Time / Compute Time Ratio")
    plt.title("Relative Ratio of Data Transfer vs. Compute Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
