import tensorflow as tf
import numpy as np

tf.config.run_functions_eagerly(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

import tensorflow as tf

def expand_to_rank(tensor, target_rank, axis=-1):
    """Expands tensor to have `target_rank` by inserting singleton dimensions."""
    num_dims = tf.maximum(target_rank - tf.rank(tensor), 0)
    return insert_dims(tensor, num_dims, axis)

def insert_dims(tensor, num_dims, axis=-1):
    """Inserts `num_dims` dimensions of length 1 at `axis`."""
    axis = axis if axis >= 0 else tf.rank(tensor) + axis + 1
    shape = tf.shape(tensor)
    new_shape = tf.concat([shape[:axis], tf.ones([num_dims], tf.int32), shape[axis:]], 0)
    return tf.reshape(tensor, new_shape)

def tf_ls_channel_estimator(y_pilots, pilots, no):
    """
    Perform Least Squares (LS) channel estimation for an OFDM MIMO system.

    Inputs:
        y_pilots: Tensor of shape [num_rx_ant, num_streams, num_pilot_symbols], tf.complex64
        pilots: Tensor of shape [num_streams, num_pilot_symbols], tf.complex64
        no: Tensor of shape [num_rx_ant], tf.float32

    Computes:
        LS channel estimate: h_ls = y_pilots / pilots
        Error variance: err_var = no / |pilots|^2

    Returns:
        h_ls: Tensor of the same shape as y_pilots, tf.complex64
        err_var: Tensor of the same shape as y_pilots, tf.float32
            Note: this is the channel estimation error variance
    """
    # Compute LS channel estimates: h_ls = y_pilots / pilots
    h_ls = tf.math.divide_no_nan(y_pilots, pilots)

    # Expand `no` for proper broadcasting to match `h_ls`
    no = expand_to_rank(no, tf.rank(h_ls), -1)

    # Expand `pilots` for proper broadcasting to match `h_ls`
    pilots = expand_to_rank(pilots, tf.rank(h_ls), 0)

    # Compute error variance: err_var = no / |pilots|^2
    err_var = tf.math.divide_no_nan(no, tf.abs(pilots) ** 2)

    return h_ls, err_var
