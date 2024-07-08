#include "cnncpp/ops.hpp"
#include <array>
#include <memory>
#include <numeric>

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

const cnncpp::Tensor<float>* cnncpp::convolution::output() const
{
    return _output_tensor.get();
}

const cnncpp::Tensor<float>* cnncpp::convolution::operator()(const Tensor<float>& input) const
{
    auto input_rows = input.dims[0];
    auto input_cols = input.dims[1];
    int d = 0;

    for (size_t kernel_depth = 0; kernel_depth < _kernels.size(); kernel_depth++) {
        const auto& kernel = _kernels.at(kernel_depth);
        for (int row = 0; row < _output_tensor->dims[0]; row++) {
            for (int col = 0; col < _output_tensor->dims[1]; col++) {
                float channel_res = 0.0;
                for (size_t channel = 0; channel < input.dims[2]; channel++) {
                    float kernel_res = 0.0;
                    for (int kernel_row = 0; kernel_row < kernel.rows; kernel_row++) {

                        auto input_roi_row_start = row * input_cols + col * _stride + kernel_row * input_cols;
                        std::vector<float> input_row(&input.data()[input_roi_row_start], &input.data()[input_roi_row_start] + kernel.cols);
                        std::vector<float> kernel_row_data(&kernel.data[kernel_row * kernel.cols], &kernel.data[(kernel_row + 1) * kernel.cols]);

                        channel_res += std::inner_product(input_row.begin(), input_row.end(), kernel_row_data.begin(), kernel_res);
                    }
                    channel_res += kernel_res;
                }
                _output_tensor->set(row, col, kernel_depth, channel_res);
            }
        }
    }
    return _output_tensor.get();
}
