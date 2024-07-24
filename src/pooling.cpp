#include "cnncpp/layers/pooling.hpp"
#include <algorithm>
#include <array>
#include <memory>
#include <numeric>

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
        for (size_t col = 0; col < input.dims[1] - _kernel_size + 1; col += _stride) {
            for (size_t row = 0; row < input.dims[0] - _kernel_size + 1; row += _stride) {
                //auto max = std::max_element(input.roi2d_iterator(row, col, depth, _kernel_size), input.roi2d_end());
                auto roi = input.create_roi({row, row+_kernel_size},
                {col, col+_kernel_size},
                {depth, depth + 1});
                auto max = std::max_element(roi.begin(), roi.end());
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
        for (size_t col = 0; col < input.dims[1] - _kernel_size + 1; col += _stride) {
            for (size_t row = 0; row < input.dims[0] - _kernel_size + 1; row += _stride) {
                auto roi = input.create_roi({row, row+_kernel_size},
                    {col, col+_kernel_size},
                    {depth, depth + 1});

                float sum = std::accumulate(roi.begin(), roi.end(), 0.0);
                _output_tensor->set(row / _stride, col / _stride, depth, sum / (_kernel_size * _kernel_size));
            }
        }
    }
    return _output_tensor.get();
}
