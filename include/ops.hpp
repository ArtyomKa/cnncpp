#ifndef _CNN_CPP_OPS_HPP_
#define _CNN_CPP_OPS_HPP_
#include "tensor.hpp"
#include <array>
#include <iterator>
#include <vector>
namespace cnncpp {

struct kernel {
    const size_t rows { 0 };
    const size_t cols { 0 };
    const std::vector<double> data;
    kernel(size_t rows, size_t cols, const std::vector<double>& data)
        : rows(rows)
        , cols(cols)
        , data(data) {};
};

class ops {

public:
    virtual const Tensor<double>* operator()(const Tensor<double>* input) const = 0;
};

class convolution : public ops {
private:
    kernel _kernel;

public:
    convolution(size_t kernel_size, size_t kernel_depth);
    virtual const Tensor<double>* operator()(const Tensor<double>* input) const;
};

}
#endif
