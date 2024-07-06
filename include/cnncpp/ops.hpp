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
    std::vector<double> data;
    kernel(size_t rows, size_t cols)
        : kernel(rows, cols, {})
    {
    }
    kernel(size_t rows, size_t cols, const std::vector<double>& data)
        : rows(rows)
        , cols(cols)
        , data(data) {};
};

class ops {

public:
    virtual const Tensor<double>* operator()(const Tensor<double>& input) const = 0;
    virtual const Tensor<double>* output() const = 0;
};

class convolution : public ops {
private:
    std::vector<kernel> _kernels;
    const size_t _stride;
    std::unique_ptr<Tensor<double>> _output_tensor;

public:
    convolution(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride, size_t kernel_depth);
    convolution(const std::array<int, 3>& input_shape, size_t kernel_size, size_t stride, size_t kernel_depth, const std::vector<double>& kernel_data);
    virtual const Tensor<double>* operator()(const Tensor<double>& input) const override;
    virtual const Tensor<double>* output() const override;
};

}
#endif
