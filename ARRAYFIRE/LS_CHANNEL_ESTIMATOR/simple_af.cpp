#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <stdexcept>

namespace py = pybind11;

/*
  af_ls_channel_estimator:
    Implements LS channel estimation for the model
      y = h * p + n,
    where h is the unknown channel, p are the known pilot symbols, and n is AWGN.
    
    For the 3D case:
      - y is a 3D np.complex64 array of shape 
            [num_rx_ant, num_streams, num_pilot_symbols],
      - p (pilots) is a 2D np.complex64 array of shape [num_streams, num_pilot_symbols],
      - no is a 1D np.float32 array of length num_rx_ant.
    
    The estimator computes:
      h_ls    = y / p     (with pilots broadcast to each rx antenna)
      err_var = no / (|p|^2) (with no expanded to all symbols)
    
    For non-3D inputs (i.e. 1D or 2D), it falls back to the previous behavior.
    
    All inputs are assumed to be in column-major (Fortran) order.
    The function returns a Python tuple (h_ls, err_var) with the same shape as y.
*/
py::tuple af_ls_channel_estimator(py::array_t<std::complex<float>> y_in,
                                  py::array_t<std::complex<float>> p_in,
                                  py::array_t<float> no_in)
{
    // Get buffer info for inputs.
    auto buf_y = y_in.request();
    auto buf_p = p_in.request();
    auto buf_no = no_in.request();

    // If y is 3D, perform batched (per rx antenna) LS estimation.
    if (buf_y.ndim == 3)
    {
        // Expected dimensions:
        //   y: [num_rx_ant, num_streams, num_pilot_symbols]
        //   p: [num_streams, num_pilot_symbols]
        //   no: [num_rx_ant]
        int num_rx_ant = buf_y.shape[0];
        int num_streams = buf_y.shape[1];
        int num_pilot_symbols = buf_y.shape[2];

        if (buf_p.ndim != 2)
            throw std::runtime_error("For 3D y, pilots must be a 2D array.");
        if (buf_p.shape[0] != num_streams || buf_p.shape[1] != num_pilot_symbols)
            throw std::runtime_error("Dimension mismatch: pilots shape must be [num_streams, num_pilot_symbols].");

        if (buf_no.ndim != 1)
            throw std::runtime_error("For 3D y, noise variance must be a 1D array.");
        if (buf_no.shape[0] != num_rx_ant)
            throw std::runtime_error("Dimension mismatch: length of noise variance must equal num_rx_ant.");

        // Create ArrayFire arrays.
        std::complex<float>* ptr_y = static_cast<std::complex<float>*>(buf_y.ptr);
        std::complex<float>* ptr_p = static_cast<std::complex<float>*>(buf_p.ptr);
        float* ptr_no = static_cast<float*>(buf_no.ptr);

        af::dim4 dims_y(num_rx_ant, num_streams, num_pilot_symbols);
        af::array y_arr(dims_y, reinterpret_cast<af::cfloat*>(ptr_y));

        af::dim4 dims_p(num_streams, num_pilot_symbols);
        af::array p_arr(dims_p, reinterpret_cast<af::cfloat*>(ptr_p));

        af::dim4 dims_no(num_rx_ant);
        af::array no_arr(dims_no, ptr_no, afHost);

        // Expand pilots to 3D by first reshaping to (1, num_streams, num_pilot_symbols)
        // and then tiling along the first dimension (rx antennas).
        af::array p_reshaped = af::moddims(p_arr, af::dim4(1, num_streams, num_pilot_symbols));
        af::array p_3d = af::tile(p_reshaped, num_rx_ant, 1, 1);

        // LS channel estimate: element-wise division.
        af::array h_ls = y_arr / p_3d;

        // Compute pilot power: |p|^2.
        af::array p_abs = af::abs(p_arr);
        af::array p_abs2 = p_abs * p_abs;
        // Expand pilot power to 3D.
        af::array p_abs2_reshaped = af::moddims(p_abs2, af::dim4(1, num_streams, num_pilot_symbols));
        af::array p_abs2_3d = af::tile(p_abs2_reshaped, num_rx_ant, 1, 1);

        // Expand noise variance: reshape to (num_rx_ant, 1, 1) and tile along the other dims.
        af::array no_reshaped = af::moddims(no_arr, af::dim4(num_rx_ant, 1, 1));
        af::array no_3d = af::tile(no_reshaped, 1, num_streams, num_pilot_symbols);

        // Compute error variance.
        af::array err_var = no_3d / p_abs2_3d;

        // Copy results to NumPy arrays.
        py::array_t<std::complex<float>> h_ls_out({num_rx_ant, num_streams, num_pilot_symbols});
        auto buf_h_ls = h_ls_out.request();
        std::complex<float>* ptr_h_ls = static_cast<std::complex<float>*>(buf_h_ls.ptr);
        h_ls.host(reinterpret_cast<af::cfloat*>(ptr_h_ls));

        py::array_t<float> err_var_out({num_rx_ant, num_streams, num_pilot_symbols});
        auto buf_err = err_var_out.request();
        float* ptr_err = static_cast<float*>(buf_err.ptr);
        err_var.host(ptr_err);

        return py::make_tuple(h_ls_out, err_var_out);
    }
    else // fallback for 1D or 2D inputs (prev code for single-sample channel estimation)
    {
        if (buf_y.ndim < 1 || buf_y.ndim > 2)
            throw std::runtime_error("y must be 1D, 2D, or 3D.");
        if (buf_p.ndim < 1 || buf_p.ndim > 2)
            throw std::runtime_error("p (pilots) must be 1D or 2D.");

        // Ensure y and p have the same shape.
        if (buf_y.shape[0] != buf_p.shape[0])
            throw std::runtime_error("Dimension mismatch: y and p must have the same size in the first dimension.");
        int P = buf_y.shape[0];
        int T = (buf_y.ndim == 2) ? buf_y.shape[1] : 1;
        if (buf_p.ndim == 2 && buf_p.shape[1] != T)
            throw std::runtime_error("Dimension mismatch: y and p must have the same shape.");

        if (buf_no.ndim > 1)
            throw std::runtime_error("Noise variance no must be a scalar.");
        float no_val = *(static_cast<float*>(buf_no.ptr));

        std::complex<float>* ptr_y = static_cast<std::complex<float>*>(buf_y.ptr);
        std::complex<float>* ptr_p = static_cast<std::complex<float>*>(buf_p.ptr);

        af::dim4 dims(P, T);
        af::array y_arr(dims, reinterpret_cast<af::cfloat*>(ptr_y));
        af::array p_arr(dims, reinterpret_cast<af::cfloat*>(ptr_p));
        af::array no_arr = af::constant(no_val, dims, f32);

        af::array h_ls = y_arr / p_arr;
        af::array p_abs = af::abs(p_arr);
        af::array p_abs2 = p_abs * p_abs;
        af::array err_var = no_arr / p_abs2;

        py::array_t<std::complex<float>> h_ls_out({P, T});
        auto buf_h_ls = h_ls_out.request();
        std::complex<float>* ptr_h_ls = static_cast<std::complex<float>*>(buf_h_ls.ptr);
        h_ls.host(reinterpret_cast<af::cfloat*>(ptr_h_ls));

        py::array_t<float> err_var_out({P, T});
        auto buf_err = err_var_out.request();
        float* ptr_err = static_cast<float*>(buf_err.ptr);
        err_var.host(ptr_err);

        return py::make_tuple(h_ls_out, err_var_out);
    }
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
    m.doc() = "ArrayFire LS Channel Estimator module supporting 3D y_pilots (batched over rx antennas)";
    m.def("af_ls_channel_estimator", &af_ls_channel_estimator,
          "Perform LS channel estimation:\n"
          "  For a 3D y (np.complex64) of shape [num_rx_ant, num_streams, num_pilot_symbols],\n"
          "  pilots (np.complex64) must be of shape [num_streams, num_pilot_symbols] and\n"
          "  noise variance no (np.float32) a 1D array of length num_rx_ant.\n"
          "  Computes h_ls = y / pilots (with pilots broadcast) and\n"
          "  err_var = no / (|pilots|^2) (with no expanded accordingly).\n"
          "If y is 1D or 2D, the original behavior is applied.\n"
          "Inputs must be in column-major (Fortran) order.",
          py::arg("y"), py::arg("p"), py::arg("no"));
    force_complex_instantiate();
}

// g++ -O2 -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) simple_af.cpp \
    -o simple_af$(python3-config --extension-suffix) \
    -I/mnt/c/Users/caden/OneDrive/Documents/ECE_113DW_Files/Sionna_Plus_ArrayFire_Setup/arrayfire/include \
    -L/mnt/c/Users/caden/OneDrive/Documents/ECE_113DW_Files/Sionna_Plus_ArrayFire_Setup/arrayfire/lib64 \
    -lafcuda
