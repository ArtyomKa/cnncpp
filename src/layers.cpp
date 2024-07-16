#include "cnncpp/layers.hpp"
#include <algorithm>
#include <array>
#include <memory>
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

const cnncpp::Tensor<float>& cnncpp::convolution::operator()(const Tensor<float>& input) const
{
    const size_t kernel_depth = input.dims[2];
    const size_t total_kernel_size = _kernel_size * _kernel_size * kernel_depth;
    for (int row = 0; row < input.dims[0] - _kernel_size + 1; row += _stride) {
        for (int col = 0; col < input.dims[1] - _kernel_size + 1; col += _stride) {
            for (size_t f = 0; f < _filters; f++) {
                auto& kernel = _kernels.at(f);
                auto val = std::inner_product(kernel.begin(), kernel.end(), input.roi_iterator(row, col, _kernel_size), 0.0f);
                auto after_activation = _activation(val + _bias[f]);
                _output_tensor->set(row / _stride, col / _stride, f, after_activation);
            }
        }
    }
    return *_output_tensor.get();
}

/*********************************************************************************
 * max_pooling
 *********************************************************************************/
cnncpp::max_pool::max_pool(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride)
    : _stride(stride)
    , _kernel_size(kernel_size)
{
    auto outd0 = (input_shape[0] - kernel_size) / _stride + 1;
    auto outd1 = (input_shape[1] - kernel_size) / _stride + 1;
    _output_tensor = std::make_unique<Tensor<float>>(outd0, outd1, input_shape[2]);
}

const cnncpp::Tensor<float>& cnncpp::max_pool::operator()(const Tensor<float>& input) const
{
    for (size_t depth = 0; depth < input.dims[2]; depth++) {
        for (int col = 0; col < input.dims[1] - _kernel_size + 1; col += _stride) {
            for (int row = 0; row < input.dims[0] - _kernel_size + 1; row += _stride) {
                auto max = std::max_element(input.roi2d_iterator(row, col, depth, _kernel_size), input.roi2d_end());
                _output_tensor->set(row / _stride, col / _stride, depth, *max);
            }
        }
    }
    return *_output_tensor.get();
}
/*********************************************************************************
 * max_pooling
 *********************************************************************************/
cnncpp::avg_pool::avg_pool(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride)
    : _stride(stride)
    , _kernel_size(kernel_size)
{
    auto outd0 = (input_shape[0] - kernel_size) / _stride + 1;
    auto outd1 = (input_shape[1] - kernel_size) / _stride + 1;
    _output_tensor = std::make_unique<Tensor<float>>(outd0, outd1, input_shape[2]);
}

const cnncpp::Tensor<float>& cnncpp::avg_pool::operator()(const Tensor<float>& input) const
{
    for (size_t depth = 0; depth < input.dims[2]; depth++) {
        for (int col = 0; col < input.dims[1] - _kernel_size + 1; col += _stride) {
            for (int row = 0; row < input.dims[0] - _kernel_size + 1; row += _stride) {
                float sum = std::accumulate(input.roi2d_iterator(row, col, depth, _kernel_size), input.roi2d_end(), 0.0);
                _output_tensor->set(row / _stride, col / _stride, depth, sum / (_kernel_size * _kernel_size));
            }
        }
    }
    return *_output_tensor.get();
}
/*********************************************************************************
 * fully_connected
 *********************************************************************************/
cnncpp::fully_connected::fully_connected(size_t input_size, size_t output_size,
    activations::activation_func_ptr activation,
    const std::vector<float>& weights,
    const std::vector<float>& bias)
    : _weights(weights)
    , _bias(bias)
    , _activation(activation)
{
    _output_tensor = std::make_unique<Tensor<float>>(output_size, 1, 1);
}

const cnncpp::Tensor<float>& cnncpp::fully_connected::operator()(const Tensor<float>& input) const
{
    for (int i = 0; i < _output_tensor->dims[0]; i++) {
        std::vector<float> weights_row(_weights.begin() + i * input.dims[0], _weights.begin() + (i + 1) * input.dims[0]);
        float val = std::inner_product(weights_row.begin(), weights_row.end(), &input.data()[0], 0.0f);
        val = _activation(val + _bias[i]);
        _output_tensor->set(i, 0, 0, val);
    }
    return *_output_tensor.get();
}

/*********************************************************************************
 * flatten
 *********************************************************************************/
cnncpp::flatten::flatten(const std::array<int, 3>& input_shape)
{
    size_t out_rows = input_shape[0] * input_shape[1] * input_shape[2];
    _output_tensor = std::make_unique<Tensor<float>>(out_rows, 1, 1);
}

const cnncpp::Tensor<float>& cnncpp::flatten::operator()(const Tensor<float>& input) const
{
    input.copyto(*_output_tensor.get());
    return *_output_tensor.get();
}
