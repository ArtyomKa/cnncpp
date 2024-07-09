#include "cnncpp/layers.hpp"
#include <algorithm>
#include <array>
#include <memory>
#include <numeric>

const cnncpp::Tensor<float>* cnncpp::layer::output() const
{
    return _output_tensor.get();
}
cnncpp::convolution::convolution(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride, size_t kernel_depth, const std::vector<float>& kernel_data)
    : _stride(stride)
{
    auto kernel_data_iterator = kernel_data.begin();
    for (int i = 0; i < kernel_depth; i++) {
        _kernels.push_back(kernel(kernel_size, kernel_size, std::vector<float>(kernel_data_iterator, kernel_data_iterator + kernel_size * kernel_size)));
        kernel_data_iterator += kernel_size * kernel_size;
    }
    auto outd0 = (input_shape[0] - _kernels[0].rows) / _stride + 1;
    auto outd1 = (input_shape[1] - _kernels[0].cols) / _stride + 1;
    _output_tensor = std::make_unique<Tensor<float>>(outd0, outd1, kernel_depth);
}
cnncpp::convolution::convolution(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride, size_t kernel_depth)
    : _kernels(kernel_depth, kernel(kernel_size, kernel_size))
    , _stride(stride)
{
    auto outd0 = (input_shape[0] - _kernels[0].rows) / _stride + 1;
    auto outd1 = (input_shape[1] - _kernels[0].cols) / _stride + 1;
    _output_tensor = std::make_unique<Tensor<float>>(outd0, outd1, kernel_depth);
}

const cnncpp::Tensor<float>* cnncpp::convolution::operator()(const Tensor<float>& input) const
{
    for (size_t kernel_depth = 0; kernel_depth < _kernels.size(); kernel_depth++) {
        const auto& kernel = _kernels.at(kernel_depth);
        for (int row = 0; row < _output_tensor->dims[0]; row++) {
            for (int col = 0; col < _output_tensor->dims[1]; col++) {
                float channel_res = 0.0;
                for (size_t channel = 0; channel < input.dims[2]; channel++) {
                    channel_res += std::inner_product(kernel.data.begin(), kernel.data.end(), input.roi_iterator(row, col, channel, kernel.rows), 0.0f);
                }
                _output_tensor->set(row, col, kernel_depth, channel_res);
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
    std::cout << "Creating max_pool layer kernel size " << kernel_size << " stride " << stride << "\n";
    auto outd0 = (input_shape[0] - kernel_size) / _stride + 1;
    auto outd1 = (input_shape[1] - kernel_size) / _stride + 1;
    _output_tensor = std::make_unique<Tensor<float>>(outd0, outd1, input_shape[2]);
}

const cnncpp::Tensor<float>* cnncpp::max_pool::operator()(const Tensor<float>& input) const
{
    for (size_t depth = 0; depth < input.dims[2]; depth++) {
        for (int col = 0; col < input.dims[1]; col += _stride) {
            for (int row = 0; row < input.dims[0]; row += _stride) {
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
        for (int col = 0; col < input.dims[1]; col += _stride) {
            for (int row = 0; row < input.dims[0]; row += _stride) {
                float sum = std::accumulate(input.roi_iterator(row, col, depth, _kernel_size), input.roi_end(), 0.0);
                _output_tensor->set(row / _stride, col / _stride, depth, sum / (_kernel_size * _kernel_size));
            }
        }
    }
    return _output_tensor.get();
}
