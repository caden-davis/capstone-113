import tensorflow as tf
import numpy as np

def tf_ofdm_demodulator(inputs, fft_size, l_min, cyclic_prefix_length):
    """
    Demodulates an OFDM waveform with cyclic prefix removal, FFT, phase compensation,
    and FFT shifting (DC in the middle).

    Parameters
    ----------
    inputs : tf.Tensor (tf.complex64) containing the time-domain signal. 
        Note: its last dimension should have length num_symbols*(fft_size + cyclic_prefix_length)
    fft_size : int for the FFT size / number of OFDM subcarriers)
    l_min : int for the largest negative time lag of the discrete-time channel impulse response (≤ 0).
    cyclic_prefix_length : int

    Returns
    -------
    tf.Tensor (tf.complex64) for the demodulated OFDM grid with shape [..., num_symbols, fft_size].
    """
    # Compute phase compensation factor: exp(-j*2*pi*k*l_min/fft_size)
    tmp = -2 * np.pi * tf.cast(l_min, tf.float32) / tf.cast(fft_size, tf.float32) \
          * tf.cast(tf.range(fft_size), tf.float32)
    phase_comp = tf.exp(tf.complex(0.0, tmp))

    # Check if cyclic_prefix_length is scalar (one CP for all symbols) or vector (different CP per symbol)
    if np.isscalar(cyclic_prefix_length):
        cp_len = int(cyclic_prefix_length)
        symbol_length = fft_size + cp_len
        total_length = tf.shape(inputs)[-1]
        # Compute number of full OFDM symbols and remove trailing samples (if any)
        num_symbols = total_length // symbol_length
        rest = tf.math.floormod(total_length, symbol_length)
        if rest != 0:
            inputs = inputs[..., :-rest]
        # Reshape to [..., num_symbols, symbol_length]
        new_shape = tf.concat([tf.shape(inputs)[:-1], [num_symbols, symbol_length]], axis=0)
        symbols = tf.reshape(inputs, new_shape)
        # Remove cyclic prefix from each symbol
        symbols = symbols[..., cp_len:cp_len+fft_size]
    else:
        # If cyclic_prefix_length is a vector (1D array-like)
        cp_vec = tf.convert_to_tensor(cyclic_prefix_length, dtype=tf.int32)
        num_symbols = cp_vec.shape[0]
        # For each symbol, total length = cp + fft_size
        row_lengths = cp_vec + fft_size
        offsets = tf.concat([tf.constant([0], dtype=tf.int32),
                             tf.cumsum(row_lengths[:-1])], axis=0)
        offsets = tf.expand_dims(offsets, 1)
        indices = tf.range(fft_size, dtype=tf.int32)
        indices = tf.reshape(indices, (1, -1))             # shape (1, fft_size)
        cp_vec = tf.reshape(cp_vec, (-1, 1))                 # shape (num_symbols, 1)
        indices = indices + cp_vec + offsets               # shape (num_symbols, fft_size)
        symbols = tf.gather(inputs, indices, axis=-1)

    # Compute FFT along the last dimension
    symbols_fft = tf.signal.fft(symbols)
    # Instead of using tf.rank (which returns a tensor), we use len() on the shape.
    rank = len(symbols_fft.shape)
    phase_comp = tf.reshape(phase_comp, [1]*(rank-1) + [fft_size])
    symbols_fft = symbols_fft * phase_comp
    # Shift DC subcarrier to the center.
    symbols_fft = tf.signal.fftshift(symbols_fft, axes=-1)
    return symbols_fft
