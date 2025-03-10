#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <stdexcept>

namespace py = pybind11;

// Global variables to hold intermediate ArrayFire arrays.
namespace {
    af::array global_y, global_h, global_s;
    af::array global_x_hat, global_noise_eff;
    int global_K;  // number of transmit antennas (from h)
}

void upload_data(py::array_t<std::complex<float>> y_in,
                 py::array_t<std::complex<float>> h_in,
                 py::array_t<std::complex<float>> s_in)
{
    auto buf_y = y_in.request();
    auto buf_h = h_in.request();
    auto buf_s = s_in.request();

    // Get dimensions.
    int M = buf_y.shape[0];
    int T = (buf_y.ndim == 1) ? 1 : buf_y.shape[1];
    int M_h = buf_h.shape[0];
    int K = buf_h.shape[1];
    int M_s = buf_s.shape[0];
    int N_s = buf_s.shape[1];
    if (M != M_h || M != M_s || M_s != N_s)
        throw std::runtime_error("Dimension mismatch among y, h, and s.");

    std::complex<float>* ptr_y = static_cast<std::complex<float>*>(buf_y.ptr);
    std::complex<float>* ptr_h = static_cast<std::complex<float>*>(buf_h.ptr);
    std::complex<float>* ptr_s = static_cast<std::complex<float>*>(buf_s.ptr);

    af::dim4 dims_y(M, T);
    af::dim4 dims_h(M, K);
    af::dim4 dims_s(M, M);

    // Create ArrayFire arrays from the host data.
    global_y = af::array(dims_y, reinterpret_cast<af::cfloat*>(ptr_y));
    global_h = af::array(dims_h, reinterpret_cast<af::cfloat*>(ptr_h));
    global_s = af::array(dims_s, reinterpret_cast<af::cfloat*>(ptr_s));
    global_K = K;
}

void compute_equalizer() {
    // Compute H * H^H.
    af::array HHh = af::matmul(global_h, global_h, AF_MAT_NONE, AF_MAT_CTRANS);
    // Form A = H*H^H + S.
    af::array A_mat = HHh + global_s;
    // Invert A.
    af::array A_inv = af::inverse(A_mat);
    // Compute G = H^H * A_inv.
    af::array G = af::matmul(global_h, A_inv, AF_MAT_CTRANS, AF_MAT_NONE);
    // Compute Gy = G * y.
    af::array Gy = af::matmul(G, global_y);
    // Compute GH = G * h.
    af::array GH = af::matmul(G, global_h);
    // Extract diagonal of GH.
    af::array d = af::diag(GH);
    // Reshape and tile d to match Gy dimensions.
    af::array d_col = af::moddims(d, af::dim4(global_K, 1));
    int T = global_y.dims(1);
    af::array d_tile = af::tile(d_col, 1, T);
    // Compute x_hat = Gy / d_tile.
    global_x_hat = Gy / d_tile;
    // Compute effective noise variance: no_eff = real(1/d - 1).
    af::array one = af::constant(1.0f, d.dims(), d.type());
    af::array inv_d = one / d;
    global_noise_eff = af::real(inv_d - one);

    af::sync(); // Force eval
}

py::tuple download_data() {
    int K = global_x_hat.dims(0);
    int T = global_x_hat.dims(1);
    // Prepare NumPy arrays for output.
    py::array_t<std::complex<float>> x_hat_out({K, T});
    auto buf_x = x_hat_out.request();
    std::complex<float>* ptr_x = static_cast<std::complex<float>*>(buf_x.ptr);
    global_x_hat.host(reinterpret_cast<af::cfloat*>(ptr_x));

    py::array_t<float> no_eff_out({K});
    auto buf_no = no_eff_out.request();
    float* ptr_no = static_cast<float*>(buf_no.ptr);
    global_noise_eff.host(ptr_no);

    return py::make_tuple(x_hat_out, no_eff_out);
}

PYBIND11_MODULE(simple_af, m) {
    m.doc() = "Split ArrayFire LMMSE equalizer functions: upload, compute, and download.";
    m.def("upload_data", &upload_data, "Upload data to GPU.");
    m.def("compute_equalizer", &compute_equalizer, "Compute the equalizer on the GPU.");
    m.def("download_data", &download_data, "Download results from GPU to CPU.");
}


// g++ -O3 -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) simple_af.cpp \
    -o simple_af$(python3-config --extension-suffix) \
    -I/mnt/c/Users/caden/OneDrive/Documents/ECE_113DW_Files/Sionna_Plus_ArrayFire_Setup/arrayfire/include \
    -L/mnt/c/Users/caden/OneDrive/Documents/ECE_113DW_Files/Sionna_Plus_ArrayFire_Setup/arrayfire/lib64 \
    -lafcuda


