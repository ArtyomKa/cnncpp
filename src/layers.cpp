#include "cnncpp/layers.hpp"
#include <algorithm>
#include <array>
#include <memory>
#include <numeric>

const cnncpp::Tensor<float>* cnncpp::layer::output() const
{
    return _output_tensor.get();
}
cnncpp::convolution::convolution(const std::array<int, 3>& input_shape,
    size_t kernel_size, size_t stride, size_t number_of_kernels,
    activations::activation_func_ptr activation,
    const std::vector<float>& kernel_data,
    const std::vector<float>& bias)
    : _stride(stride)
    , _activation(activation)

    , _bias(bias)
{
    std::vector<float>::const_iterator kernel_data_iterator = kernel_data.cbegin();
    const auto kernel_total = kernel_size * kernel_size * input_shape[2];
    for (int i = 0; i < number_of_kernels; i++) {
        std::vector<float> kernel_(kernel_data.cbegin() + kernel_total * i, kernel_data.cbegin() + kernel_total * (i + 1));
        _kernels.push_back(kernel(kernel_size, kernel_size, input_shape[2], kernel_));
        kernel_data_iterator += kernel_size * kernel_size;
    }
    auto outd0 = (input_shape[0] - _kernels[0].rows) / _stride + 1;
    auto outd1 = (input_shape[1] - _kernels[0].cols) / _stride + 1;
    _output_tensor = std::make_unique<Tensor<float>>(outd0, outd1, number_of_kernels);
}
cnncpp::convolution::convolution(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride, size_t number_of_kernels)
    : _kernels(number_of_kernels, kernel(kernel_size, kernel_size, input_shape[2]))
    , _stride(stride)
{
    auto outd0 = (input_shape[0] - _kernels[0].rows) / _stride + 1;
    auto outd1 = (input_shape[1] - _kernels[0].cols) / _stride + 1;
    _output_tensor = std::make_unique<Tensor<float>>(outd0, outd1, number_of_kernels);
}

const cnncpp::Tensor<float>* cnncpp::convolution::operator()(const Tensor<float>& input) const
{
    for (size_t kernel_num = 0; kernel_num < _kernels.size(); kernel_num++) {
        const auto& kernel = _kernels.at(kernel_num);
        const auto kernel_size = kernel.rows * kernel.cols;
        for (int row = 0; row < input.dims[0] - kernel.rows + 1; row += _stride) {
            for (int col = 0; col < input.dims[1] - kernel.cols + 1; col += _stride) {
                float channel_res = 0.0;
                for (size_t channel = 0; channel < input.dims[2]; channel++) {
                    channel_res += std::inner_product(kernel.data.begin() + channel * kernel_size, kernel.data.begin() + (channel + 1) * kernel_size, input.roi_iterator(row, col, channel, kernel.rows), 0.0f);
                }
                auto val = _activation(channel_res + _bias[kernel_num]);
                _output_tensor->set(row / _stride, col / _stride, kernel_num, val);
            }
        }
    }
    return _output_tensor.get();
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

const cnncpp::Tensor<float>* cnncpp::max_pool::operator()(const Tensor<float>& input) const
{
    for (size_t depth = 0; depth < input.dims[2]; depth++) {
        for (int col = 0; col < input.dims[1] - _kernel_size + 1; col += _stride) {
            for (int row = 0; row < input.dims[0] - _kernel_size + 1; row += _stride) {
                // float sum = std::accumulate(input.roi_iterator(row, col, depth, _kernel_size), input.roi_end(), 0.0);
                auto max = std::max_element(input.roi_iterator(row, col, depth, _kernel_size), input.roi_end());
                _output_tensor->set(row / _stride, col / _stride, depth, *max);
            }
        }
    }
    return _output_tensor.get();
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

const cnncpp::Tensor<float>* cnncpp::avg_pool::operator()(const Tensor<float>& input) const
{
    for (size_t depth = 0; depth < input.dims[2]; depth++) {
        for (int col = 0; col < input.dims[1] - _kernel_size + 1; col += _stride) {
            for (int row = 0; row < input.dims[0] - _kernel_size + 1; row += _stride) {
                float sum = std::accumulate(input.roi_iterator(row, col, depth, _kernel_size), input.roi_end(), 0.0);
                _output_tensor->set(row / _stride, col / _stride, depth, sum / (_kernel_size * _kernel_size));
            }
        }
    }
    return _output_tensor.get();
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

const cnncpp::Tensor<float>* cnncpp::fully_connected::operator()(const Tensor<float>& input) const
{
    for (int i = 0; i < _output_tensor->dims[0]; i++) {
        std::vector<float> weights_row(_weights.begin() + i * input.dims[0], _weights.begin() + (i + 1) * input.dims[0]);
        float val = std::inner_product(weights_row.begin(), weights_row.end(), &input.data()[0], 0.0f);
        val = _activation(val + _bias[i]);
        _output_tensor->set(i, 0, 0, val);
    }
    return _output_tensor.get();
}

/*********************************************************************************
 * flatten
 *********************************************************************************/
cnncpp::flatten::flatten(const std::array<int, 3>& input_shape)
{
    size_t out_rows = input_shape[0] * input_shape[1] * input_shape[2];
    _output_tensor = std::make_unique<Tensor<float>>(out_rows, 1, 1);
}
const cnncpp::Tensor<float>* cnncpp::flatten::operator()(const Tensor<float>& input) const
{

    input.copyto(*_output_tensor.get());

    for (int row = 0; row < input.dims[0]; row++) {
        for (int col = 0; col < input.dims[1]; col++) {
            for (int channel = 0; channel < input.dims[2]; channel++) {
                float val = *input.roi_iterator(row, col, channel, 1);
                _output_tensor->set(channel + col * input.dims[2] + row * input.dims[2] * input.dims[1], 0, 0, val);
            }
        }
    }
    return _output_tensor.get();
}
