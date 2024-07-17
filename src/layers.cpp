#include "cnncpp/layers/layers.hpp"
#include <array>
#include <memory>
#include <numeric>

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
