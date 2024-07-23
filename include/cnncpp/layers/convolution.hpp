#ifndef __CNNCPP_LAYERS_CONVOLUTION_HPP__
#define __CNNCPP_LAYERS_CONVOLUTION_HPP__
#include "cnncpp/activations.hpp"
#include "cnncpp/layers/layers.hpp"
#include "cnncpp/utils.hpp"
#include <array>
#include <cstddef>
#include <vector>
namespace cnncpp {

class convolution : public layer {
private:
    std::vector<std::vector<float>> _kernels;
    std::vector<float> _bias;
    const stride_t _stride;
    const kernel_size_t _kernel_size;
    const filters_t _filters;
    activations::activation_func_ptr _activation;

public:
    convolution(const std::array<int, 3>& input_shape, kernel_size_t kernel_size, stride_t stride, filters_t filters);
    convolution(const std::array<int, 3>& input_shape,
        size_t kernel_size, size_t stride, size_t filters,
        activations::activation_func_ptr activation,
        const std::vector<float>& kernel_data,
        const std::vector<float>& bias);
    virtual const Tensor<float>* operator()(const Tensor<float>& input) const override;
};

}
#endif
