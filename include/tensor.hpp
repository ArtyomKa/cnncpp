#ifndef _CNN_CPP_TENSOR_HPP_
#define _CNN_CPP_TENSOR_HPP_

#include <array>
#include <cstddef>
#include <iostream>
#include <vector>
namespace cnncpp {
template <typename T>
class Tensor {

private:
    std::vector<T> _data;

public:
    const std::array<std::size_t, 3> dims;
    constexpr Tensor(size_t rows, size_t cols, size_t depth)
        : dims { rows, cols, depth }
        , _data(rows * cols * depth) {};
    constexpr Tensor(size_t rows, size_t cols, size_t depth, const std::vector<T>& data)
        : dims { rows, cols, depth }
        , _data(data)
    {
        if (rows * cols * depth != data.size()) {
            throw std::length_error("input data is incompatible with supplied dimensions");
        }
        std::cout << "Copy Tensor"
                  << "\n";
    };

    const T* data() const
    {
        return _data.data();
    }

    int total() const
    {
        return dims[0] * dims[1] * dims[2];
    }
};
} // namespace
#endif // _CNN_CPP_TENSOR_HPP_
