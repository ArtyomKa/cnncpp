#include "gtest/gtest.h"

#include "cnncpp/layers.hpp"
#include "tests.hpp"
TEST(MaxPooling, OutputDims)
{
    cnncpp::max_pool max_pool_layer({ 28, 28, 6 }, 2, 2);
    auto output = max_pool_layer.output();
    ASSERT_EQ(14, output->dims[0]);
    ASSERT_EQ(14, output->dims[1]);
    ASSERT_EQ(6, output->dims[2]);
}
TEST(MaxPooling, MaxPool2x2)
{
    cnncpp::Tensor<float> input(2, 2, 1, { 0.0, 0.1, 0.7, 0.3 });
    cnncpp::max_pool max_pool_layer({ 2, 2, 1 }, 2, 1);

    auto output = max_pool_layer(input);
    ASSERT_EQ(1, output->dims[0]);
    ASSERT_EQ(1, output->dims[1]);
    ASSERT_EQ(1, output->dims[2]);
    ASSERT_NEAR(output->data()[0], 0.7, EPSILON);
}

TEST(MaxPooling, MaxPool4x4input2x2filer)
{
    cnncpp::Tensor<float> input(4, 4, 1, { 0.0, 0.1, 0.7, 0.3,
                                             /////////////////
                                             0.1, -0.2, 0.1, 0.5,
                                             /////////////////
                                             0.3, -0.5, 1.0, 0.2,
                                             /////////////////
                                             -1.0, 0.0, 2.0, 1.0 });
    cnncpp::max_pool max_pool_layer({ 4, 4, 1 }, 2, 2);

    auto output = max_pool_layer(input);
    ASSERT_EQ(2, output->dims[0]);
    ASSERT_EQ(2, output->dims[1]);
    ASSERT_EQ(1, output->dims[2]);
    ASSERT_NEAR(output->data()[0], 0.1, EPSILON);
    ASSERT_NEAR(output->data()[1], 0.7, EPSILON);
    ASSERT_NEAR(output->data()[2], 0.3, EPSILON);
    ASSERT_NEAR(output->data()[3], 2.0, EPSILON);
}
TEST(AvgPooling, OutputDims)
{
    cnncpp::avg_pool avg_pool_layer({ 28, 28, 6 }, 2, 2);
    auto output = avg_pool_layer.output();
    ASSERT_EQ(14, output->dims[0]);
    ASSERT_EQ(14, output->dims[1]);
    ASSERT_EQ(6, output->dims[2]);
}
TEST(AvgPooling, AvgPool2x2)
{
    cnncpp::Tensor<float> input(2, 2, 1, { 0.0, 0.1, 0.7, 0.3 });
    cnncpp::avg_pool avg_pool_layer({ 2, 2, 1 }, 2, 1);

    auto output = avg_pool_layer(input);
    ASSERT_EQ(1, output->dims[0]);
    ASSERT_EQ(1, output->dims[1]);
    ASSERT_EQ(1, output->dims[2]);
    ASSERT_NEAR(output->data()[0], (0.7 + 0.0 + 0.1 + 0.3) / 4, EPSILON);
}

TEST(AvgPooling, AvgPool4x4input2x2filer)
{
    cnncpp::Tensor<float> input(4, 4, 1, { 0.0, 0.1, 0.7, 0.3,
                                             /////////////////
                                             0.1, -0.2, 0.1, 0.5,
                                             /////////////////
                                             0.3, -0.5, 1.0, 0.2,
                                             /////////////////
                                             -1.0, 0.0, 2.0, 1.0 });
    cnncpp::avg_pool avg_pool_layer({ 4, 4, 1 }, 2, 2);

    auto output = avg_pool_layer(input);
    ASSERT_EQ(2, output->dims[0]);
    ASSERT_EQ(2, output->dims[1]);
    ASSERT_EQ(1, output->dims[2]);
    ASSERT_NEAR(output->data()[0], (0.1 + 0.0 + 0.1 - 0.2) / 4, EPSILON);
    ASSERT_NEAR(output->data()[1], (0.7 + 0.3 + 0.1 + 0.5) / 4, EPSILON);
    ASSERT_NEAR(output->data()[2], (0.3 - 0.5 - 1.0 + 0) / 4, EPSILON);
    ASSERT_NEAR(output->data()[3], (1.0 + 0.2 + 2.0 + 1.0) / 4, EPSILON);
}
