#include "cnncpp/tensor.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

static constexpr float EPSILON = 1e-5;

TEST(TensorTest, BasicCreate)
{
    // basic creation of an empty tensor with given dimensions
    cnncpp::Tensor<float> tensor(3, 3, 3);
    EXPECT_EQ(tensor.dims[0], 3);
    EXPECT_EQ(tensor.dims[1], 3);
    EXPECT_EQ(tensor.dims[2], 3);
}
TEST(TensorTest, CreateWithGivenData)
{
    // basic creation of a tensor with given dimensions and data
    std::vector<float> data { 0.0, 0.1, 0.2, 0.3 };
    cnncpp::Tensor<float> tensor(2, 2, 1, data);
    EXPECT_EQ(tensor.dims[0], 2);
    EXPECT_EQ(tensor.dims[1], 2);
    EXPECT_EQ(tensor.dims[2], 1);
    const auto& _data = tensor.data();
    EXPECT_TRUE(std::equal(data.begin(), data.end(), &_data[0]));
}

TEST(TensorTest, CreateWithGivenDataIncompatibleDims)
{
    // basic creation of a tensor with incompatible dimensions and data
    std::vector<float> data { 0.0, 0.1, 0.2 };
    EXPECT_THROW(cnncpp::Tensor<float> tensor(2, 2, 1, data), std::length_error);
}

TEST(TensorTest, CreateROIIterator)
{
    std::vector<float> data { 0.0, 0.1, 0.2, 0.3 };
    cnncpp::Tensor<float> tensor(2, 2, 1, data);
    auto iter1 = tensor.roi_iterator(0, 0, 0, 1);
    ASSERT_EQ(*iter1, 0.0);
    auto iter2 = tensor.roi_iterator(0, 1, 0, 1);
    ASSERT_NEAR(*iter2, 0.1, EPSILON);
}

TEST(TensorTest, TraverseOverROIEqualsToFullDimensions)
{
    std::vector<float> data { 0.0, 0.1, 0.2, 0.3 };
    cnncpp::Tensor<float> tensor(2, 2, 1, data);
    auto iter1 = tensor.roi_iterator(0, 0, 0, 2);
    ASSERT_EQ(*iter1, 0.0);
    ASSERT_NEAR(*(++iter1), 0.1, EPSILON);
}
TEST(TensorTest, TraverseOverROI)
{
    std::vector<float> data {
        0.0, 0.1, 0.2,
        0.3, 0.4, 0.5,
        0.6, 0.7, 0.8
    };
    cnncpp::Tensor<float> tensor(3, 3, 1, data);
    auto iter1 = tensor.roi_iterator(1, 1, 0, 2);
    std::vector<float> test { 0.3, 0.4, 0.6, 0.7 };
    for (auto iter = tensor.roi_iterator(1, 0, 0, 2); iter != tensor.roi_end(); ++iter)
        std::cout << *iter << " ";
    ASSERT_TRUE(std::equal(tensor.roi_iterator(1, 0, 0, 2), tensor.roi_end(), test.begin()));
}
