#ifndef _CNN_CPP_OPS_HPP_
#define _CNN_CPP_OPS_HPP_
#include "activations.hpp"
#include "tensor.hpp"
#include <array>
#include <cstddef>
#include <iterator>
#include <memory>
#include <vector>
namespace cnncpp {

class layer {

protected:
    std::unique_ptr<Tensor<float>> _output_tensor;

public:
    virtual const Tensor<float>& operator()(const Tensor<float>& input) const = 0;
    const Tensor<float>& output() const;
};

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

class max_pool : public layer {
private:
    const size_t _stride;
    const size_t _kernel_size;

public:
    max_pool(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride);
    virtual const Tensor<float>& operator()(const Tensor<float>& input) const override;
};
class avg_pool : public layer {
private:
    const size_t _stride;
    const size_t _kernel_size;

public:
    avg_pool(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride);
    virtual const Tensor<float>& operator()(const Tensor<float>& input) const override;
};

class fully_connected : public layer {
private:
    std::vector<float> _weights;
    std::vector<float> _bias;
    activations::activation_func_ptr _activation;

public:
    fully_connected(size_t input_size, size_t output_size, activations::activation_func_ptr activation, const std::vector<float>& weights, const std::vector<float>& bias);
    virtual const Tensor<float>& operator()(const Tensor<float>& input) const override;
};

class flatten : public layer {
public:
    flatten(const std::array<int, 3>& input_shape);
    virtual const Tensor<float>& operator()(const Tensor<float>& input) const override;
};
}
#endif
