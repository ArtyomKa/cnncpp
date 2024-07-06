#include "cnncpp/tensor.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

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
