#include "cnncpp/layers/layers.hpp"
#include "tests.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(FullyConneected, OuputCreatedTransposition)
{
    cnncpp::fully_connected fc_layer(4, 3, cnncpp::activations::none,
        { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11, 12 }, { 0.0, 0.0, 0.0 });
    const auto& output = fc_layer.output();
    ASSERT_EQ(output.dims[0], 3);
    ASSERT_EQ(output.dims[1], 1);
    ASSERT_EQ(output.dims[2], 1);
}

TEST(FullyConneected, OuputCreatedWithCorrectDimentions)
{
    cnncpp::fully_connected fc_layer(400, 120, cnncpp::activations::none,
        std::vector<float>(400 * 120, 1.0), std::vector<float>(120, 0.1));
    const auto& output = fc_layer.output();
    ASSERT_EQ(output.dims[0], 120);
    ASSERT_EQ(output.dims[1], 1);
    ASSERT_EQ(output.dims[2], 1);
}

TEST(FullyConneected, OuputComputeWithBias)
{
    std::vector<float> weights { 1, 2, 3,
        -4, -5, -6,
        7, 8, 9,
        -10, -11, -12 };
    cnncpp::fully_connected fc_layer(4, 3, cnncpp::activations::none, weights, { 0, 0.3, -0.5 });
    cnncpp::Tensor<float> input(4, 1, 1, { 1.0, 2.0, 3.0, 4.0f });
    auto output = fc_layer(input);
    ASSERT_EQ(output->dims[0], 3);
    ASSERT_EQ(output->dims[1], 1);
    ASSERT_EQ(output->dims[2], 1);

    ASSERT_NEAR(1 - 4 * 2.0 + 7 * 3.0 - 10 * 4, output->data()[0], EPSILON);
    ASSERT_NEAR(1 * 2 - 5 * 2 + 8 * 3 - 11 * 4 + 0.3, output->data()[1], EPSILON);
    ASSERT_NEAR(3 * 1 - 6 * 2 + 9 * 3 - 12 * 4 - 0.5, output->data()[2], EPSILON);
}
