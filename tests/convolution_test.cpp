#include "cnncpp/ops.hpp"
#include "gtest/gtest.h"
#include <numeric>
#include <vector>

TEST(ConvolutionTest, VectorConvolution)
{
    std::vector<double> _window = { 0.6, 0.1, 0.6 };
    std::vector<double> _kernel = { -1, -1, -1 };
    auto result = std::inner_product(_window.cbegin(), _window.cend(), _kernel.cbegin(), 0.0);
    ASSERT_EQ(result, -0.6 + (-0.1) + (-0.6));
}

TEST(ConvolutionTest, OutputTensorIsCreatedWithCorrectDims)
{
    cnncpp::convolution conv({ 227, 227, 3 }, 11, 4, 96);
    auto output_tensor = conv.output();
    ASSERT_EQ(output_tensor->dims[0], 55);
    ASSERT_EQ(output_tensor->dims[1], 55);
    ASSERT_EQ(output_tensor->dims[2], 96);
}

TEST(ConvolutionTest, Simple2DConvolution)
{
    cnncpp::Tensor<double> input(3, 3, 1, { 0.6, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9 });
    cnncpp::convolution conv({ 3, 3, 1 }, 3, 1, 1, { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
    auto res = conv(input);
    ASSERT_EQ(res->dims[0], 1);
    ASSERT_EQ(res->dims[1], 1);
    ASSERT_EQ(res->dims[2], 1);
}
TEST(ConvolutionTest, 4x42DConvolution)
{
    cnncpp::Tensor<double> input(4, 4, 1, { 0.6, 1.0, 0.6, 0.2, 1.0, 1.0, 1.0, 0.9, 1.0, 0.9, 0.9, 1.0, 1.0, 0.8, 0.3, 1.0 });
    cnncpp::convolution conv({ 4, 4, 1 }, 3, 1, 1, { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 });
    auto res = conv(input);
    ASSERT_EQ(res->dims[0], 2);
    ASSERT_EQ(res->dims[1], 2);
    ASSERT_EQ(res->dims[2], 1);
}
