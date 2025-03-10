#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

/*
  af_lmmse_equalizer:
    Implements a MIMO LMMSE equalizer for the model
      y = H x + n,  with E[n n^H] = S,
    where the transmit symbols are zero-mean with identity covariance.
    
    The equalizer is given by:
      G = H^H * inv( H*H^H + S )
      x_hat = diag(G * H)^{-1} * (G * y)
      no_eff = real( 1/diag(G*H) - 1 )
      
    Here, we now assume:
      y: a 2D np.complex64 array of shape (M, T), where M is the number of receive antennas,
         and T is the number of symbols (or channel uses).
      h: a 2D np.complex64 array of shape (M, K) (channel matrix)
      s: a 2D np.complex64 array of shape (M, M) (noise covariance)
      
    The inputs must be provided in column-major (Fortran) order.
    Intermediate arrays are printed using print().
    The function returns a Python tuple (x_hat, no_eff), where:
      - x_hat is a (K, T) complex array (equalized symbols for each channel use),
      - no_eff is a (K,) float vector (effective noise variance per symbol).
*/
py::tuple af_lmmse_equalizer(py::array_t<std::complex<float>> y_in,
                             py::array_t<std::complex<float>> h_in,
                             py::array_t<std::complex<float>> s_in)
{
    // Get buffer info.
    auto buf_y = y_in.request();
    auto buf_h = h_in.request();
    auto buf_s = s_in.request();
    if (buf_y.ndim != 2 && buf_y.ndim != 1)
        throw std::runtime_error("y must be 1D or 2D; for a multi-symbol signal, use 2D with shape (M, T)");
    if (buf_h.ndim != 2)
        throw std::runtime_error("h must be 2D");
    if (buf_s.ndim != 2)
        throw std::runtime_error("s must be 2D");

    // Let M be number of receive antennas.
    int M = buf_y.shape[0];
    // If y is 1D, we treat it as (M, 1); otherwise, T is the number of symbols.
    int T = (buf_y.ndim == 1) ? 1 : buf_y.shape[1];
    int M_h = buf_h.shape[0]; // should equal M
    int K = buf_h.shape[1];   // number of transmit antennas
    int M_s = buf_s.shape[0]; // s must be M x M
    int N_s = buf_s.shape[1];
    if (M != M_h || M != M_s || M_s != N_s)
        throw std::runtime_error("Dimension mismatch: y must have length M (or shape (M, T)), h must be (M, K), and s must be (M, M)");

    // Create ArrayFire arrays from input pointers.
    std::complex<float>* ptr_y = static_cast<std::complex<float>*>(buf_y.ptr);
    std::complex<float>* ptr_h = static_cast<std::complex<float>*>(buf_h.ptr);
    std::complex<float>* ptr_s = static_cast<std::complex<float>*>(buf_s.ptr);

    // y_arr: shape (M, T)
    af::dim4 dims_y(M, T);
    af::array y_arr(dims_y, reinterpret_cast<af::cfloat*>(ptr_y));

    // h_arr: shape (M, K)
    af::dim4 dims_h(M, K);
    af::array h_arr(dims_h, reinterpret_cast<af::cfloat*>(ptr_h));

    // s_arr: shape (M, M)
    af::dim4 dims_s(M, M);
    af::array s_arr(dims_s, reinterpret_cast<af::cfloat*>(ptr_s));

    // Print inputs.
    // print("y_arr:", y_arr);
    // print("h_arr:", h_arr);
    // print("s_arr:", s_arr);

    // Step 1: Compute H * H^H.
    // Use af::matmul with AF_MAT_CTRANS for conjugate transpose.
    af::array HHh = af::matmul(h_arr, h_arr, AF_MAT_NONE, AF_MAT_CTRANS);
    // print("H * H^H:", HHh);

    // Step 2: Form A = H*H^H + S.
    af::array A_mat = HHh + s_arr;
    // print("A = H*H^H + S:", A_mat);

    // Step 3: Invert A.
    af::array A_inv = af::inverse(A_mat);
    // print("A_inv:", A_inv);

    // Step 4: Compute G = H^H * A_inv.
    af::array G = af::matmul(h_arr, A_inv, AF_MAT_CTRANS, AF_MAT_NONE);
    // print("G = H^H * inv(A):", G);

    // Step 5: Compute Gy = G * y.
    // G has shape (K, M) and y_arr is (M, T) so Gy is (K, T).
    af::array Gy = af::matmul(G, y_arr);
    // print("Gy = G * y:", Gy);

    // Step 6: Compute GH = G * h.
    // G (K, M) multiplied by h_arr (M, K) yields (K, K).
    af::array GH = af::matmul(G, h_arr);
    // print("GH = G * h:", GH);

    // Step 7: Extract diagonal of GH.
    af::array d = af::diag(GH);
    // print("diag(GH):", d);

    // Step 8: Compute x_hat = Gy / d.
    // Since d is a vector of length K, we need to expand its dimensions to divide Gy (K, T).
    af::array d_col = af::moddims(d, af::dim4(K, 1));  // shape (K,1)
    // Broadcasting: tile d_col along dimension 1 T times.
    af::array d_tile = af::tile(d_col, 1, T);            // shape (K, T)
    af::array x_hat = Gy / d_tile;
    // print("x_hat = Gy / diag(GH):", x_hat);

    // Step 9: Compute effective noise variance: no_eff = real(1/d - 1).
    af::array one = af::constant(1.0f, d.dims(), d.type());
    af::array inv_d = one / d;
    af::array noise_eff = af::real(inv_d - one);
    // print("no_eff = real(1/d - 1):", noise_eff);

    // x_hat is (K, T) and noise_eff is (K,); we output them as such.
    // Copy results into NumPy arrays.
    py::array_t<std::complex<float>> x_hat_out({K, T});
    auto buf_x = x_hat_out.request();
    std::complex<float>* ptr_x = static_cast<std::complex<float>*>(buf_x.ptr);
    x_hat.host(reinterpret_cast<af::cfloat*>(ptr_x));

    py::array_t<float> no_eff_out({K});
    auto buf_no = no_eff_out.request();
    float* ptr_no = static_cast<float*>(buf_no.ptr);
    noise_eff.host(ptr_no);

    return py::make_tuple(x_hat_out, no_eff_out);
}

// Helper to force instantiation of ArrayFire’s complex routines.
namespace {
    void force_complex_instantiate() {
        af::dim4 dims(1);
        std::complex<float> dummy(0, 0);
        af::cfloat dummy_af = *reinterpret_cast<af::cfloat*>(&dummy);
        af::array arr(dims, &dummy_af);
        (void)arr;
    }
}

PYBIND11_MODULE(simple_af, m) {
    m.doc() = "ArrayFire LMMSE MIMO equalizer module (whiten_interference=False, supports y with extra symbol dimension)";
    m.def("af_lmmse_equalizer", &af_lmmse_equalizer,
          "Perform LMMSE equalization: given received signal y (shape (M,T)), channel matrix h (shape (M,K)), and noise covariance s (shape (M,M)),\n"
          "compute x_hat = diag(GH)^{-1} * (G * y) and effective noise variance no_eff (vector of length K).\n"
          "Inputs must be np.complex64 arrays in column-major (Fortran) order. h and s are assumed constant over symbols.",
          py::arg("y"), py::arg("h"), py::arg("s"));
    force_complex_instantiate();
}



// g++ -O2 -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) simple_af.cpp \
    -o simple_af$(python3-config --extension-suffix) \
    -I/mnt/c/Users/caden/OneDrive/Documents/ECE_113DW_Files/Sionna_Plus_ArrayFire_Setup/arrayfire/include \
    -L/mnt/c/Users/caden/OneDrive/Documents/ECE_113DW_Files/Sionna_Plus_ArrayFire_Setup/arrayfire/lib64 \
    -lafcuda
