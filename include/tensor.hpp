#ifndef _CNN_CPP_TENSOR_HPP_
#define _CNN_CPP_TENSOR_HPP_

#include <array>
#include <cstddef>
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
    constexpr Tensor(size_t rows, size_t cols, size_t depth, const std::vector<T> data))
	: dims { rows, cols, depth }, _data(data) {
	std::cout << "Copy Tensor" << "\n" 

        };
};
} // namespace
#endif // _CNN_CPP_TENSOR_HPP_
