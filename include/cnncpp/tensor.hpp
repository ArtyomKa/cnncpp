#ifndef _CNN_CPP_TENSOR_HPP_
#define _CNN_CPP_TENSOR_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <vector>
namespace cnncpp {
// Main data storage class. The data stored in a vector in column major order.
template <typename T>
class Tensor {

public:
    // iterator class for iterating over an ROI - added so the convolution on a specific ROI can be implemented as std::inner_product with the kernel data.
    // used u=in the convolution layer
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
        size_t _roi_row { 0 };
        size_t _roi_col { 0 };
        size_t _current_row { 0 };
        size_t _current_col { 0 };
        size_t _current_depth { 0 };
        bool _end { true };

    public:
        iterator()
            : _end(true) {
            };
        iterator(const Tensor<T>& tensor, int row, int col, int roi_size)
            : _rows(tensor.dims[0])
            , _cols(tensor.dims[1])
            , _depth(tensor.dims[2])
            , _roi_size(roi_size)
            , _roi_row(row)
            , _roi_col(col)
            , _current_row(0)
            , _current_col(0)
            , _end(false)

        {
            _current = (tensor._data.begin() + row * _cols * _depth + col * _depth);
        }
        iterator& operator++() noexcept
        {
            if (_end) {
                return *this;
            }
            // if (_current_row + 1 == _roi_size && _current_col + 1 == _roi_size) {
            if (_current_depth + 1 < _depth) {
                _current_depth++;
                ++_current;
                // _end = true;
                return *this;
            } else {
                _current_depth = 0;
            }
            if (_current_col + 1 < _roi_size) {
                // if (_current_col + 1 < _roi_size) {
                _current_col++;
                ++_current;
                return *this;
            } else {
                _current_col = 0;
            }
            if (_current_row + 1 < _roi_size) {
                _current_row++;
                _current += ((_cols - _roi_size) * _depth + 1);
                return *this;
            } else {
                _end = true;
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

    // 2d iterator class - similar to the iterator above but iterates over 2d ROI on a specific depth of the tensor.
    // Used in pooling layers (max and average pooling)
    class iterator2d {
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
        size_t _roi_row { 0 };
        size_t _roi_col { 0 };
        size_t _current_row { 0 };
        size_t _current_col { 0 };
        size_t _current_depth { 0 };
        bool _end { true };

    public:
        iterator2d()
            : _end(true) {
            };
        iterator2d(const Tensor<T>& tensor, int row, int col, int channel, int roi_size)
            : _rows(tensor.dims[0])
            , _cols(tensor.dims[1])
            , _depth(tensor.dims[2])
            , _roi_size(roi_size)
            , _roi_row(row)
            , _roi_col(col)
            , _current_row(0)
            , _current_col(0)
            , _end(false)

        {
            _current = (tensor._data.begin() + row * _cols * _depth + col * _depth + channel);
        }
        iterator2d& operator++() noexcept
        {
            if (_end) {
                return *this;
            }
            // if (_current_row + 1 == _roi_size && _current_col + 1 == _roi_size) {
            if (_current_col + 1 < _roi_size) {
                // if (_current_col + 1 < _roi_size) {
                _current_col++;
                _current += _depth;
                return *this;
            } else {
                _current_col = 0;
            }
            if (_current_row + 1 < _roi_size) {
                _current_row++;
                _current += ((_cols - _roi_size + 1) * _depth);
                return *this;
            } else {
                _end = true;
                return *this;
            }
            std::cout << "Error";

            return *this;
        }
        iterator2d operator++(int)
        {
            iterator2d temp = *this;
            ++_current;
            return temp;
        }
        // iterator& operator=(const iterator& other) = default;
        value_type operator*() const { return *_current; };
        pointer operator->() const { return _current; }
        friend bool operator==(const iterator2d& lhs, const iterator2d& rhs)
        {
            if (lhs._end && rhs._end)
                return true;
            return lhs._current == rhs._current;
        }
        friend bool operator!=(const iterator2d& lhs, const iterator2d& rhs)
        {
            return !(lhs == rhs);
        }
    };

private:
    std::vector<T> _data;

public:
    std::vector<T>& data_vec()
    {
        return _data;
    }
    const std::vector<T>& data_vec() const
    {
        return _data;
    }
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
        auto index = depth + col * dims[2] + row * dims[1] * dims[2];
        assert(index < _data.size());
        _data[index] = val;
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
    iterator roi_iterator(int row, int col, int roi_size) const
    {
        return iterator(*this, row, col, roi_size);
    }
    iterator roi_end() const
    {
        return iterator();
    }
    iterator2d roi2d_iterator(int row, int col, int depth, int roi_size) const
    {
        return iterator2d(*this, row, col, depth, roi_size);
    }
    iterator2d roi2d_end() const
    {
        return iterator2d();
    }
    Tensor(const Tensor<T>& other) = delete;
    const Tensor<T>& operator=(const Tensor<T>& other) = delete;
    Tensor(Tensor<T>&& other) = delete;
    const Tensor<T>& operator=(Tensor<T>&& other) = delete;
};

typedef Tensor<float> TensorF;

static_assert(std::forward_iterator<cnncpp::Tensor<float>::iterator>);

} // namespace
#endif // _CNN_CPP_TENSOR_HPP_
