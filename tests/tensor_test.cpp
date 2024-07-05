#include "tensor.hpp"
#include "gtest/gtest.h"
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
    // basic creation of an empty tensor with given dimensions
    std::vector<float> data { 0.0, 0.1, 0.2, 0.3 };
    cnncpp::Tensor<float> tensor(2, 2, 1, data.data());
    EXPECT_EQ(tensor.dims[0], 3);
    EXPECT_EQ(tensor.dims[1], 3);
    EXPECT_EQ(tensor.dims[2], 3);
}
