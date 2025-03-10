import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import simple_af      # The compiled ArrayFire module
import simple_tf      # The TensorFlow combined module

def af_to_tf(array):
    """
    Converts an ArrayFire output array from Fortran (column-major) ordering to TensorFlow C (row-major) ordering.
    """
    # First, flatten the array in C order (i.e. the order AF actually returned it)
    flat = np.ravel(array, order='C')
    # Then rebuild an array of the same shape by filling it in Fortran order.
    # This “reshuffles” the elements into the order TensorFlow expects.
    return np.reshape(flat, array.shape, order='F')

def benchmark_af_list(y_list, pilots_list, no_list):
    times = []
    results = []
    total_list = list(zip(y_list, pilots_list, no_list))
    
    # Warm-up using the first input set (if available)
    if len(y_list) > 0:
        y, pilots, no = total_list[0]
        y_f      = np.asfortranarray(y)
        pilots_f = np.asfortranarray(pilots)
        no_f     = np.asfortranarray(no)
        af_h_ls, af_err_var = simple_af.af_ls_channel_estimator(y_f, pilots_f, no_f)
    
    for (y, pilots, no) in total_list:
        y_f      = np.asfortranarray(y)
        pilots_f = np.asfortranarray(pilots)
        no_f     = np.asfortranarray(no)
        start = time.perf_counter()
        af_h_ls, af_err_var = simple_af.af_ls_channel_estimator(y_f, pilots_f, no_f)
        end = time.perf_counter()
        times.append(end - start)
        results.append((af_h_ls, af_err_var))
        
    avg_time = sum(times) / len(times)
    return avg_time, results

def benchmark_tf_list(y_list, pilots_list, no_list):
    times = []
    results = []
    total_list = list(zip(y_list, pilots_list, no_list))
    
    # Warm-up using the first input set (if available)
    if len(y_list) > 0:
        y, pilots, no = total_list[0]
        y_f      = np.asfortranarray(y)
        pilots_f = np.asfortranarray(pilots)
        no_f     = np.asfortranarray(no)
        tf_h_ls, tf_err_var = simple_tf.tf_ls_channel_estimator(
            tf.convert_to_tensor(y_f),
            tf.convert_to_tensor(pilots_f),
            tf.convert_to_tensor(no_f)
        )
    
    for (y, pilots, no) in total_list:
        y_f      = np.asfortranarray(y)
        pilots_f = np.asfortranarray(pilots)
        no_f     = np.asfortranarray(no)
        start = time.perf_counter()
        tf_h_ls, tf_err_var = simple_tf.tf_ls_channel_estimator(
            tf.convert_to_tensor(y_f),
            tf.convert_to_tensor(pilots_f),
            tf.convert_to_tensor(no_f)
        )
        # Convert TensorFlow tensors to NumPy arrays.
        tf_h_ls = tf_h_ls.numpy()
        tf_err_var = tf_err_var.numpy()
        end = time.perf_counter()
        times.append(end - start)
        results.append((tf_h_ls, tf_err_var))
    
    avg_time = sum(times) / len(times)
    return avg_time, results

def main():
    # K is the number of transmit antennas/streams for each datapoint
    Ks = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 768, 1024]
    num_runs = 5  # Number of random input sets per size
    T = 20        # Number of pilot symbols

    # For plotting speedup: speedup = (TF average time) / (AF average time)
    speedups = []

    for K in Ks:
        num_streams = K
        num_rx_ant = 2 * K  # Twice as many receiver antennas as transmitter antennas

        y_list = []
        pilots_list = []
        no_list = []
        
        for _ in range(num_runs):
            # Generate random complex inputs.
            # y_pilots: shape (num_rx_ant, num_streams, T)
            y_pilots = (np.random.randn(num_rx_ant, num_streams, T) +
                        1j * np.random.randn(num_rx_ant, num_streams, T)).astype(np.complex64)
            # pilots: shape (num_streams, T)
            pilots = (np.random.randn(num_streams, T) +
                      1j * np.random.randn(num_streams, T)).astype(np.complex64)
            # no: noise variance per receiver antenna, shape (num_rx_ant,)
            no = np.abs(np.random.randn(num_rx_ant)).astype(np.float32)
            
            y_list.append(y_pilots)
            pilots_list.append(pilots)
            no_list.append(no)
        
        print(f"=== Benchmarking LS Channel Estimator for {num_streams} streams, {num_rx_ant} Rx antennas, {T} pilot symbols ===")
        
        # Benchmark TensorFlow LS Channel Estimator.
        print("=== TensorFlow LS Channel Estimator ===")
        tf_time, tf_results = benchmark_tf_list(y_list, pilots_list, no_list)
        
        # Benchmark ArrayFire LS Channel Estimator.
        print("=== ArrayFire LS Channel Estimator ===")
        af_time, af_results = benchmark_af_list(y_list, pilots_list, no_list)
        
        # Compute speedup: how many times faster is ArrayFire relative to TensorFlow?
        speedup = tf_time / max(af_time, 1e-6)
        print(f"AF over TF Speedup: {speedup:.6f} x")
        speedups.append(speedup)
        
        # Compare the outputs run-by-run.
        all_match = True
        for i, (r_tf, r_af) in enumerate(zip(tf_results, af_results)):
            tf_h_ls, tf_err_var = r_tf
            af_h_ls, af_err_var = r_af
            # Convert ArrayFire outputs to TensorFlow’s ordering.
            af_h_ls_conv = af_to_tf(af_h_ls)
            af_err_var_conv = af_to_tf(af_err_var)
            if not (np.allclose(tf_h_ls, af_h_ls_conv, rtol=1e-3, atol=1e-3) and
                    np.allclose(tf_err_var, af_err_var_conv, rtol=1e-3, atol=1e-3)):
                if not (np.allclose(tf_h_ls, af_h_ls_conv, rtol=1e-1, atol=1e-1) and
                        np.allclose(tf_err_var, af_err_var_conv, rtol=1e-1, atol=1e-1)):
                    print(f"Run {i}: The LS channel estimation results differ!")
                    all_match = False
                else:
                    print(f"Run {i}: The LS channel estimation results should be inspected more closely.")
        if all_match:
            print(f"All LS channel estimation results match for {num_streams} streams!")
        else:
            print(f"Some LS channel estimation results differ for {num_streams} streams; please inspect further.")
        
        # Print a sample 3x3 submatrix of the LS estimates from run 0 for visual inspection.
        # print("ArrayFire LS estimates (3x3 submatrix) from run 0:")
        af_h_ls_sample = af_to_tf(af_results[0][0])
        # print(af_h_ls_sample[:3, :3])
        # print("TensorFlow LS estimates (3x3 submatrix) from run 0:")
        # print(tf_results[0][0][:3, :3])
        print("\n")
    
    # Plot speedup vs. number of streams.
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(10, 8))
    plt.plot(Ks, speedups, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Streams")
    plt.ylabel("AF over TF Speedup")
    plt.title("ArrayFire vs. TensorFlow LS Channel Estimator - RTX 2000 Ada")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
