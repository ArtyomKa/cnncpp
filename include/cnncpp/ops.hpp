#ifndef _CNN_CPP_OPS_HPP_
#define _CNN_CPP_OPS_HPP_
#include "tensor.hpp"
#include <array>
#include <iterator>
#include <memory>
#include <vector>
namespace cnncpp {

struct kernel {
    const size_t rows { 0 };
    const size_t cols { 0 };
    std::vector<float> data;
    kernel(size_t rows, size_t cols)
        : kernel(rows, cols, {})
    {
    }
    kernel(size_t rows, size_t cols, const std::vector<float>& data)
        : rows(rows)
        , cols(cols)
        , data(data) {};
};

class ops {

public:
    virtual const Tensor<float>* operator()(const Tensor<float>& input) const = 0;
    virtual const Tensor<float>* output() const = 0;
};

class convolution : public ops {
private:
    std::vector<kernel> _kernels;
    const size_t _stride;
    std::unique_ptr<Tensor<float>> _output_tensor;

public:
    convolution(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride, size_t kernel_depth);
    convolution(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride, size_t kernel_depth, const std::vector<float>& kernel_data);
    virtual const Tensor<float>* operator()(const Tensor<float>& input) const override;
    virtual const Tensor<float>* output() const override;
};

}
#endif
