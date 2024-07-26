#ifndef __CNNCPP_LAYERS_LAYERS_HPP__
#define __CNNCPP_LAYERS_LAYERS_HPP__
#include "cnncpp/activations.hpp"
#include "cnncpp/tensor.hpp"
#include <array>
#include <cstddef>
#include <memory>
#include <vector>
namespace cnncpp {

// abstract base class for layers.
class layer {

protected:
    std::unique_ptr<Tensor<float>> _output_tensor;

public:
    // performs the layer logic and returns a pointer to the result without granting ownership.
    virtual const Tensor<float>* operator()(const Tensor<float>& input) const = 0;
    const Tensor<float>& output() const;
};

class fully_connected : public layer {
private:
    std::vector<float> _weights;
    std::vector<float> _bias;
    activations::activation_func_ptr _activation;

public:
    fully_connected(size_t input_size, size_t output_size, activations::activation_func_ptr activation, const std::vector<float>& weights, const std::vector<float>& bias);
    virtual const Tensor<float>* operator()(const Tensor<float>& input) const override;
};

class flatten : public layer {
public:
    flatten(const std::array<int, 3>& input_shape);
    virtual const Tensor<float>* operator()(const Tensor<float>& input) const override;
};
}
#endif
