import tensorflow as tf
import numpy as np

tf.config.run_functions_eagerly(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

def tf_lmmse_equalizer(y, h, s):
    """
    Perform LMMSE equalization (whiten_interference=False) for the MIMO model:
       y = H x + n,  with E[n n^H] = s.
    Inputs:
      y: a tensor of shape (M,) or (M, T) with dtype tf.complex64,
         where M is the number of receive antennas and T is the number of symbols.
      h: a tensor of shape (M, K) with dtype tf.complex64.
      s: a tensor of shape (M, M) with dtype tf.complex64.
    Computes:
      G = H^H * inv(H * H^H + s)
      x_hat = diag(G * H)^{-1} * (G * y)   (applied to each symbol if T > 1)
      no_eff = real(1/diag(G*H) - 1)
    Prints intermediate results at each step.
    Returns:
      - x_hat: tensor of shape (K,) if y is (M,), or (K, T) if y is (M, T).
      - no_eff: tensor of shape (K,) with dtype tf.float32.
    """
    # print("Input y:\n", y)
    # print("Input h:\n", h)
    # print("Input s:\n", s)
    
    # Step 1: Compute H * H^H.
    HHh = tf.matmul(h, h, adjoint_b=True)
    # print("H * H^H:\n", HHh.numpy())
    
    # Step 2: Compute A = H*H^H + s.
    A = HHh + s
    # print("A = H*H^H + s:\n", A.numpy())
    
    # Step 3: Invert A.
    A_inv = tf.linalg.inv(A)
    # print("A_inv:\n", A_inv.numpy())
    
    # Step 4: Compute G = H^H * A_inv.
    G = tf.matmul(h, A_inv, adjoint_a=True)
    # print("G = H^H * inv(A):\n", G.numpy())
    
    # Step 5: Compute Gy = G * y.
    # If y is 1D, expand it to (M,1). Otherwise, assume y is (M,T).
    if len(y.shape) == 1:
        y_proc = tf.expand_dims(y, -1)  # shape (M, 1)
    else:
        y_proc = y  # assume shape (M, T)
    Gy = tf.matmul(G, y_proc)  # shape (K, T) if y is (M,T), else (K,1)
    # print("Gy = G * y:\n", Gy.numpy())
    
    # Step 6: Compute GH = G * h.
    GH = tf.matmul(G, h)  # shape (K, K)
    # print("GH = G * h:\n", GH.numpy())
    
    # Step 7: Extract diagonal of GH.
    d = tf.linalg.diag_part(GH)  # shape (K,)
    # print("diag(GH):\n", d.numpy())
    
    # Step 8: Compute x_hat = Gy / diag(GH) (element-wise division).
    # To ensure proper broadcasting, expand d to shape (K,1).
    d_expanded = tf.expand_dims(d, -1)  # shape (K,1)
    x_hat = Gy / d_expanded
    # print("x_hat = Gy / diag(GH):\n", x_hat.numpy())
    
    # Step 9: Compute effective noise variance: no_eff = real(1/d - 1).
    one = tf.constant(1.0, dtype=d.dtype)
    no_eff = tf.math.real(one/d - one)
    # print("no_eff = real(1/d - 1):\n", no_eff.numpy())
    
    return x_hat, no_eff
