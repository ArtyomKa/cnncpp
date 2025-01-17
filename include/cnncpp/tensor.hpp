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
struct range {
    const size_t begin;
    const size_t end;
};

template <typename T>
class roi : public std::ranges::view_interface<roi<T>> {
    std::vector<T>::const_iterator _begin;
    const range _rows_range;
    const range _columns_range;
    const range _depth_range;
    const size_t _rows;
    const size_t _cols;
    const size_t _channels;

public:
    class iterator {

    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = const T&;

        iterator()
            : _current(nullptr)
            , _roi(nullptr)
            , _current_row(0)
            , _current_col(0)
            , _current_channel(0)
        {
        }
        explicit iterator(roi* roi)
            : _current(nullptr)
            , _roi(roi)
            , _current_row(_roi->_rows_range.begin)
            , _current_col(_roi->_columns_range.begin)
            , _current_channel(_roi->_depth_range.begin)
        {
            set_current();
        }

        reference operator*() const
        {
            return (*_current);
        }
        const pointer operator->() { return _current; }

        iterator& operator++()
        {
            if (_current_channel < _roi->_depth_range.end - 1) {
                ++_current;
                _current_channel++;
                return *this;
            } else {
                _current_channel = _roi->_depth_range.begin;
            }
            if (_current_col < _roi->_columns_range.end - 1) {
                _current_col++;
                set_current();
                return *this;
            } else {
                _current_col = _roi->_columns_range.begin;
            }
            if (_current_row < _roi->_rows_range.end - 1) {
                _current_row++;
                set_current();
                return *this;
            } else {
                //_current = std::end();
                _roi = nullptr;
                return *this;
            }
        }
        iterator operator++(int)
        {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const iterator& a, const iterator& b)
        {
            if (a._roi == nullptr && b._roi == nullptr)
                return true; // end
            return (a._current == b._current) && (a._roi == b._roi);
        }
        friend bool operator!=(const iterator& a, const iterator& b) { return !(a == b); }

    private:
        void set_current()
        {
            _current = _roi->_begin + _current_channel + _roi->_channels * (_current_col + _roi->_cols * _current_row);
        }
        std::vector<T>::const_iterator _current;
        roi<float>* _roi;
        size_t _current_row;
        size_t _current_col;
        size_t _current_channel;
    };

public:
    roi(std::vector<T>::const_iterator begin_, size_t rows, size_t cols, size_t channels,
        const range& rrows, const range& rcols, const range& rdepth)
        : _begin(begin_)
        , _rows_range(rrows)
        , _columns_range(rcols)
        , _depth_range(rdepth)
        , _rows(rows)
        , _cols(cols)
        , _channels(channels)
    {
        
    }
    iterator begin()
    {
        return iterator(this);
    }
    iterator end()
    {
        return iterator();
    }
};

// Main data storage class. The data stored in a vector in column major order.
template <typename T>
class Tensor {

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

    roi<float> create_roi(const range& rrows, const range& rcols, const range& rdepth) const
    {
        return roi<float>(_data.cbegin(), dims[0], dims[1], dims[2], rrows, rcols, rdepth);
    }

    Tensor(const Tensor<T>& other) = delete;
    const Tensor<T>& operator=(const Tensor<T>& other) = delete;
    Tensor(Tensor<T>&& other) = delete;
    const Tensor<T>& operator=(Tensor<T>&& other) = delete;

    friend class roi<float>;
    friend class roi<float>::iterator;
};

typedef Tensor<float> TensorF;

static_assert(std::input_iterator<roi<float>::iterator>);
static_assert(std::ranges::input_range<roi<float>>);

} // namespace
#endif // _CNN_CPP_TENSOR_HPP_
