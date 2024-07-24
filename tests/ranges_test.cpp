#include "gtest/gtest.h"
#include <cstddef>
#include <iterator>
#include <numeric>
#include <ranges>
#include <vector>
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
            if (a._roi == nullptr && b._roi == nullptr) return true; //end
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
        // this->ptr += (_depth_range.begin + _channels * (_columns_range.begin + _cols * _rows_range.begin));
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
class Ctr {

    size_t _rows;
    size_t _cols;
    size_t _channels;
    std::vector<float> _data;

public:
    Ctr(size_t rows, size_t cols, size_t channels)
        : _rows(rows)
        , _cols(cols)
        , _channels(channels)
    {
    }
    Ctr(size_t rows, size_t cols, size_t channels, std::initializer_list<float> data)
        : Ctr(rows, cols, channels)
    {
        _data = data;
    }
    Ctr(size_t rows, size_t cols, size_t channels, std::vector<float>&& data)
        : Ctr(rows, cols, channels)
    {
        _data = std::move(data);
    }

    roi<float> create_roi(const range& rrows, const range& rcols, const range& rdepth) const
    {
        return roi<float>(_data.cbegin(), _rows, _cols, _channels, rrows, rcols, rdepth);
    }
    friend class roi<float>;
    friend class roi<float>::iterator;
};
auto begin(roi<float> const&);
auto end(roi<float> const&);
static_assert(std::input_iterator<roi<float>::iterator>);
static_assert(std::ranges::input_range<roi<float>>);

TEST(RangesTest, CreateROI3DIteration)
{
    std::vector<float> data(27);
    std::iota(data.begin(), data.end(), 1); // creates a box 3x3x3 running indexes (1-based)
    const Ctr t(3, 3, 3, std::move(data));
    auto roi = t.create_roi({ 1, 3 }, { 1, 3 }, { 1, 3 });
    std::vector<float> test { 14, 15, 17, 18, 23, 24, 26, 27 };
    std::for_each(roi.begin(), roi.end(), [](auto val) {std::cout << val << "\n"; });
    ASSERT_TRUE(std::ranges::equal(roi, test));
}
