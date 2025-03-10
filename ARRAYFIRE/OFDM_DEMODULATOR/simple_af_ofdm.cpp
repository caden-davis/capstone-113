#include <arrayfire.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <iostream>
#include <stdexcept>

namespace py = pybind11;

/*
  af_ofdm_demodulator:
    Demodulates an OFDM waveform with cyclic prefix removal, matching TensorFlow dimension ordering.

    Inputs:
        inputs: a 1D np.complex<float> array containing the time-domain signal
        fft_size: FFT size (number of subcarriers)
        l_min: Largest negative time lag (<= 0)
        cyclic_prefix_length: Cyclic prefix length (>= 0)
      
    Output:
        A np.complex<float> array of shape (num_symbols, fft_size), where
        dimension 0 = symbol index and dimension 1 = subcarrier index
*/
py::array_t<std::complex<float>> af_ofdm_demodulator(py::array_t<std::complex<float>> inputs,
                                                     int fft_size,
                                                     int l_min,
                                                     int cyclic_prefix_length)
{
    if (cyclic_prefix_length < 0)
        throw std::runtime_error("cyclic_prefix_length must be nonnegative.");

    // Get input buffer info (expects a 1D array).
    auto buf = inputs.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Input must be a 1D array representing the time-domain signal.");

    size_t total_length = buf.shape[0];
    int symbol_length = fft_size + cyclic_prefix_length;
    int num_symbols = total_length / symbol_length;
    size_t used_length = num_symbols * symbol_length;

    // Create an ArrayFire array from the input data (ignore trailing samples).
    std::complex<float>* ptr = static_cast<std::complex<float>*>(buf.ptr);
    af::array inp = af::array(used_length, reinterpret_cast<af::cfloat*>(ptr));

    // Reshape so that dimension 0 = num_symbols, dimension 1 = time (symbol_length).
    // That is, shape (num_symbols, symbol_length).
    inp = af::moddims(inp, num_symbols, symbol_length);

    // Remove cyclic prefix by slicing along dimension 1:
    // shape becomes (num_symbols, fft_size).
    af::array symbols = inp(af::span, af::seq(cyclic_prefix_length, cyclic_prefix_length + fft_size - 1));

    // Compute FFT along dimension 1 (the subcarrier dimension).
    // shape remains (num_symbols, fft_size).
    af::array symbols_fft = af::fft(symbols, fft_size, /*dim*/ 1);

    // Phase compensation: exp(-j*2*pi*k*l_min/fft_size), broadcast along dimension 0.
    // dimension 1 has length fft_size, so we build k as shape (1, fft_size).
    af::dim4 dims_k(1, fft_size);
    af::array k = af::range(dims_k, 1, f32);  // 1st dimension = 1, 2nd dimension = fft_size
    af::array exponent = -2.0f * af::Pi * l_min * k / fft_size;
    af::array phase_comp = af::exp(af::complex(0.0f, exponent));
    // Multiply (num_symbols, fft_size) by (1, fft_size) -> broadcast along dimension 0.
    symbols_fft = symbols_fft * phase_comp;

    // fftshift along dimension 1 by fft_size/2 to move DC to center subcarrier.
    symbols_fft = af::shift(symbols_fft, 0, fft_size/2);

    // The final shape is (num_symbols, fft_size).
    // ==> Prepare output numpy array.
    std::vector<dim_t> dims = {
        static_cast<dim_t>(num_symbols),
        static_cast<dim_t>(fft_size)
    };
    py::array_t<std::complex<float>> output(dims);
    auto buf_out = output.request();
    std::complex<float>* out_ptr = static_cast<std::complex<float>*>(buf_out.ptr);
    symbols_fft.host(reinterpret_cast<af::cfloat*>(out_ptr));

    return output;
}

// Helper to force ArrayFire's complex routines instantiation.
namespace {
    void force_complex_instantiate() {
        af::dim4 dims(1);
        std::complex<float> dummy(0, 0);
        af::cfloat dummy_af = *reinterpret_cast<af::cfloat*>(&dummy);
        af::array arr(dims, &dummy_af);
        (void)arr;
    }
}

PYBIND11_MODULE(simple_af_ofdm, m) {
    m.doc() = "ArrayFire OFDM Demodulator module (matches TensorFlow dimension ordering)";
    m.def("af_ofdm_demodulator", &af_ofdm_demodulator,
          "Demodulate an OFDM waveform with cyclic prefix removal, returning shape (num_symbols, fft_size).\n"
          "Matches TensorFlow's dimension ordering.",
          py::arg("inputs"), py::arg("fft_size"), py::arg("l_min"), py::arg("cyclic_prefix_length"));
    force_complex_instantiate();
}


// g++ -O2 -shared -std=c++14 -fPIC $(python3 -m pybind11 --includes) simple_af_ofdm.cpp \
    -o simple_af_ofdm$(python3-config --extension-suffix) \
    -I/mnt/c/Users/caden/OneDrive/Documents/ECE_113DW_Files/Sionna_Plus_ArrayFire_Setup/arrayfire/include \
    -L/mnt/c/Users/caden/OneDrive/Documents/ECE_113DW_Files/Sionna_Plus_ArrayFire_Setup/arrayfire/lib64 \
    -lafcuda

