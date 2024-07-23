#include "cnncpp/layers/convolution.hpp"
#include <array>
#include <numeric>

const cnncpp::Tensor<float>& cnncpp::layer::output() const
{
    return *_output_tensor.get();
}

cnncpp::convolution::convolution(const std::array<int, 3>& input_shape,
    size_t kernel_size, size_t stride, size_t filters,
    activations::activation_func_ptr activation,
    const std::vector<float>& kernel_data,
    const std::vector<float>& bias)
    : _stride(stride)
    , _kernel_size(kernel_size)
    , _filters(filters)
    , _activation(activation)
    , _kernels(std::vector<std::vector<float>>(filters))
    , _bias(bias)
{
    auto outd0 = (input_shape[0] - kernel_size) / _stride + 1;
    auto outd1 = (input_shape[1] - kernel_size) / _stride + 1;

    for (auto& k : _kernels) {
        k.reserve(kernel_size * kernel_size * input_shape[2]);
    }
    for (size_t f = 0; f < kernel_data.size(); f++) {
        _kernels.at(f % _filters).push_back(kernel_data[f]);
    }
    _output_tensor = std::make_unique<Tensor<float>>(outd0, outd1, filters);
}

cnncpp::convolution::convolution(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride, size_t number_of_kernels)
    : convolution(input_shape, kernel_size, stride, number_of_kernels, activations::none, {}, {})
{
}

const cnncpp::Tensor<float>* cnncpp::convolution::operator()(const Tensor<float>& input) const
{
    const size_t kernel_depth = input.dims[2];
    const size_t total_kernel_size = _kernel_size * _kernel_size * kernel_depth;
    for (int row = 0; row < input.dims[0] - _kernel_size + 1; row += _stride) {
        for (int col = 0; col < input.dims[1] - _kernel_size + 1; col += _stride) {
            for (size_t f = 0; f < _filters; f++) {
                auto& kernel = _kernels.at(f);
                auto val = std::inner_product(kernel.begin(), kernel.end(), input.roi_iterator(row, col, _kernel_size), 0.0f);
                // auto after_activation = _activation(val + _bias[f]);
                val += _bias[f];
                _output_tensor->set(row / _stride, col / _stride, f, val);
            }
        }
    }
    _activation(_output_tensor->data_vec());
    return _output_tensor.get();
}
