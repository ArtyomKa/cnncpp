
#ifndef __CNNCPP_LAYERS_POOLING__
#define __CNNCPP_LAYERS_POOLING__

#include "cnncpp/layers/layers.hpp"

namespace cnncpp {
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
}
#endif
