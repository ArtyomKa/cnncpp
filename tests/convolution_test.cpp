#include "cnncpp/layers/convolution.hpp"
#include "gtest/gtest.h"
#include <numeric>

static constexpr float EPSILON = 1e-5;
TEST(ConvolutionTest, VectorConvolution)
{
    std::vector<float> _window = { 0.6f, 0.1f, 0.6f };
    std::vector<float> _kernel = { -1., -1., -1. };
    auto result = std::inner_product(_window.cbegin(), _window.cend(), _kernel.cbegin(), 0.0);
    ASSERT_NEAR(result, -0.6 + (-0.1) + (-0.6), EPSILON);
}

TEST(ConvolutionTest, OutputTensorIsCreatedWithCorrectDims2)
{
    cnncpp::convolution conv({ 32, 32, 1 }, 5, 1, 6);
    auto &output_tensor = conv.output();
    ASSERT_EQ(output_tensor.dims[0], 28);
    ASSERT_EQ(output_tensor.dims[1], 28);
    ASSERT_EQ(output_tensor.dims[2], 6);
}

TEST(ConvolutionTest, OutputTensorIsCreatedWithCorrectDims)
{
    cnncpp::convolution conv({ 227, 227, 3 }, 11, 4, 96);
    auto &output_tensor = conv.output();
    ASSERT_EQ(output_tensor.dims[0], 55);
    ASSERT_EQ(output_tensor.dims[1], 55);
    ASSERT_EQ(output_tensor.dims[2], 96);
}

TEST(ConvolutionTest, Simple2DConvolution)
{
    std::vector<float> kernel { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 };
    cnncpp::Tensor<float> input(3, 3, 1, { 0.6, 1.0, 0.6, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9 });
    std::vector<float> weights = { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 };
    cnncpp::convolution conv({ 3, 3, 1 }, 3, 1, 1, cnncpp::activations::none, weights, std::vector<float> { .0 });
    auto res2 = std::inner_product(kernel.begin(), kernel.end(), input.create_roi({0,3}, {0,3}, {0,1}).begin(), 0.0);
    auto res = conv(input);
    ASSERT_EQ(res->dims[0], 1);
    ASSERT_EQ(res->dims[1], 1);
    ASSERT_EQ(res->dims[2], 1);
    ASSERT_NEAR(res->data()[0], 0.8, EPSILON);
}
TEST(ConvolutionTest, 4x42DConvolution)
{
    cnncpp::Tensor<float> input(4, 4, 1, { 0.6, 1.0, 0.6, 0.2, 1.0, 1.0, 1.0, 0.9, 1.0, 0.9, 0.9, 1.0, 1.0, 0.8, 0.3, 1.0 });
    cnncpp::convolution conv({ 4, 4, 1 }, 3, 1, 1, cnncpp::activations::none, { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 }, std::vector<float>(1, 0.0));
    auto res = conv(input);
    ASSERT_EQ(res->dims[0], 2);
    ASSERT_EQ(res->dims[1], 2);
    ASSERT_EQ(res->dims[2], 1);
    ASSERT_NEAR(res->data()[0], 0.8, EPSILON);
    ASSERT_NEAR(res->data()[1], 1.1, EPSILON);
    ASSERT_NEAR(res->data()[2], -0.2, EPSILON);
    ASSERT_NEAR(res->data()[3], -0.1, EPSILON);
}
TEST(ConvolutionTest, 4x42DConvolutionInputDepth3)
{
    std::vector<float> input_data(4 * 4 * 3);
    std::iota(input_data.begin(), input_data.end(), 0);
    cnncpp::Tensor<float> input(4, 4, 3, input_data);
    cnncpp::convolution conv({ 4, 4, 3 }, 3, 1, 1, cnncpp::activations::none, { -1., 0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1., -1., 0., 1. }, std::vector<float>(1, 0.0));
    auto  res = conv(input);
    ASSERT_EQ  (res->dims[0], 2);
    ASSERT_EQ  (res->dims[1], 2);
    ASSERT_EQ  (res->dims[2], 1);
    ASSERT_NEAR(res->data()[0], 18, EPSILON);
    ASSERT_NEAR(res->data()[1], 18, EPSILON);
    ASSERT_NEAR(res->data()[2], 18, EPSILON);
    ASSERT_NEAR(res->data()[3], 18, EPSILON);
}
TEST(ConvolutionTest, 4x42DConvolutionInputDepth3Kernels2)
{
    std::vector<float> input_data(4 * 4 * 3, 1.0);
    for (int i = 0; i < input_data.size(); i++) {
        input_data[i] = i % 3 + 1;
    }

    cnncpp::Tensor<float> input(4, 4, 3, input_data);
    std::vector<float> weights(3 * 3 * 3 * 2, 1); // weights are 2 kernels of 3x3x3
    // first kernel is al 1.0. second is all -1.0
    for (size_t i = 1; i < weights.size(); i += 2) {
        weights[i] = -1;
    }
    // std::fill(weights.begin(), weights.begin() + 3 * 3 * 3, 1.0);
    // std::fill(weights.begin() + 3 * 3 * 3, weights.end(), -1.0);
    std::vector<float> biases(2, 0.0); // biases length equals to the number of kernels
    cnncpp::convolution conv({ 4, 4, 3 }, 3, 1, 2, cnncpp::activations::none, weights, biases);
    auto res = conv(input);
    ASSERT_EQ(  res->dims[0], 2);
    ASSERT_EQ(  res->dims[1], 2);
    ASSERT_EQ(  res->dims[2], 2);
    ASSERT_NEAR(res->data()[0], 9 + 9 * 2 + 9 * 3, EPSILON);
    ASSERT_NEAR(res->data()[1], -9 - 9 * 2 - 9 * 3, EPSILON);
    ASSERT_NEAR(res->data()[2], 9 + 9 * 2 + 9 * 3, EPSILON);
    ASSERT_NEAR(res->data()[3], -9 - 9 * 2 - 9 * 3, EPSILON);
    ASSERT_NEAR(res->data()[4], 9 + 9 * 2 + 9 * 3, EPSILON);
    ASSERT_NEAR(res->data()[5], -9 - 9 * 2 - 9 * 3, EPSILON);
    ASSERT_NEAR(res->data()[6], 9 + 9 * 2 + 9 * 3, EPSILON);
    ASSERT_NEAR(res->data()[7], -9 - 9 * 2 - 9 * 3, EPSILON);
}
