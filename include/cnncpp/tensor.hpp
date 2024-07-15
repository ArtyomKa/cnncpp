#ifndef _CNN_CPP_TENSOR_HPP_
#define _CNN_CPP_TENSOR_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <exception>
#include <iostream>
#include <iterator>
#include <vector>
namespace cnncpp {
template <typename T>
class Tensor {

public:
    class iterator {
    public:
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;
        using iterator_category = std::forward_iterator_tag;

    private:
        std::vector<T>::const_iterator _current;
        size_t _rows { 0 };
        size_t _cols { 0 };
        size_t _depth { 0 };
        size_t _roi_size { 0 };
        size_t _current_row { 0 };
        size_t _current_col { 0 };
        bool _end { true };

    public:
        iterator()
            : _end(true) {
            };
        iterator(const Tensor<T>& tensor, int row, int col, int channel, int roi_size)
            : _rows(tensor.dims[0])
            , _cols(tensor.dims[1])
            , _depth(tensor.dims[2])
            , _roi_size(roi_size)
            , _current_row(0)
            , _current_col(0)
            , _end(false)

        {
            _current = (tensor._data.begin() + channel * _rows * _cols + row * _cols + col);
        }
        iterator& operator++() noexcept
        {
            if (_end) {
                return *this;
            }
            if (_current_row + 1 == _roi_size && _current_col + 1 == _roi_size) {
                _end = true;
                return *this;
            }
            if (_current_col + 1 < _roi_size) {
                _current_col++;
                _current++;
                return *this;
            }
            if (_current_col + 1 == _roi_size && _current_row + 1 < _roi_size) {
                _current_col = 0;
                _current_row++;
                _current += (_cols - _roi_size + 1);
                return *this;
            }
            std::cout << "Error";

            return *this;
        }
        iterator operator++(int)
        {
            iterator temp = *this;
            ++_current;
            return temp;
        }
        // iterator& operator=(const iterator& other) = default;
        value_type operator*() const { return *_current; };
        pointer operator->() const { return _current; }
        friend bool operator==(const iterator& lhs, const iterator& rhs)
        {
            if (lhs._end && rhs._end)
                return true;
            return lhs._current == rhs._current;
        }
        friend bool operator!=(const iterator& lhs, const iterator& rhs)
        {
            return !(lhs == rhs);
        }
    };

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
        if (rows * cols * depth != _data.size()) {
            throw std::length_error("input data is incompatible with supplied dimensions");
        }
    };

    constexpr Tensor(size_t rows, size_t cols, size_t depth, std::vector<T>&& data)
        : dims { rows, cols, depth }
    {
        if (rows * cols * depth != data.size()) {
            throw std::length_error("input data is incompatible with supplied dimensions");
        }
        _data = std::move(data);
    };
    const T* data() const
    {
        return _data.data();
    }
    void set(size_t row, size_t col, size_t depth, T val)
    {
        auto index = depth * dims[0] * dims[1] + row * dims[1] + col;
        assert(index < _data.size());
        _data[depth * dims[0] * dims[1] + row * dims[1] + col] = val;
    }

    void copyto(Tensor<T>& out) const
    {
        assert(_data.size() == out._data.size());
        std::copy(_data.cbegin(), _data.cend(), out._data.begin());
    }

    int total() const
    {
        return dims[0] * dims[1] * dims[2];
    }
    iterator roi_iterator(int row, int col, int depth, int roi_size) const
    {
        return iterator(*this, row, col, depth, roi_size);
    }
    iterator roi_end() const
    {
        return iterator();
    }
};

typedef Tensor<float> TensorF;

static_assert(std::forward_iterator<cnncpp::Tensor<float>::iterator>);

} // namespace
#endif // _CNN_CPP_TENSOR_HPP_
