#ifndef __CNNCPP_LAYERS_CONVOLUTION_HPP__
#define __CNNCPP_LAYERS_CONVOLUTION_HPP__
#include "cnncpp/activations.hpp"
#include "cnncpp/layers/layers.hpp"
#include <array>
#include <cstddef>
#include <vector>
namespace cnncpp {

class convolution : public layer {
private:
    std::vector<std::vector<float>> _kernels;
    std::vector<float> _bias;
    const size_t _stride;
    const size_t _kernel_size;
    const size_t _filters;
    activations::activation_func_ptr _activation;

public:
    convolution(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride, size_t number_of_kernels);
    convolution(const std::array<int, 3>& input_shape,
        size_t kernel_size, size_t stride, size_t number_of_kernels,
        activations::activation_func_ptr activation,
        const std::vector<float>& kernel_data,
        const std::vector<float>& bias);
    virtual const Tensor<float>& operator()(const Tensor<float>& input) const override;
};

}
#endif
