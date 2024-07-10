#include "cnncpp/activations.hpp"
#include "cnncpp/layers.hpp"
#include "tests.hpp"
#include <gtest/gtest.h>
#include <vector>

TEST(FullyConneected, OuputCreatedWithCorrectDimentions)
{
    cnncpp::fully_connected fc_layer(400, 120, cnncpp::activations::none,
        std::vector<float>(400 * 120, 1.0), std::vector<float>(120, 0.1));
    auto output = fc_layer.output();
    ASSERT_EQ(output->dims[0], 120);
    ASSERT_EQ(output->dims[1], 1);
    ASSERT_EQ(output->dims[2], 1);
}

TEST(FullyConneected, OuputComputeWithBias)
{
    std::vector<float> weights { 0.74864643, -1.00722027, 1.45983017, 1.34236011,
        -1.20116017, -0.08884298, -0.46555646, 0.02341039,
        -0.30973958, 0.89235565, -0.92841053, 0.12266543 };
    cnncpp::fully_connected fc_layer(4, 3, cnncpp::activations::none, weights, { 0, 0.3, -0.5 });
    cnncpp::Tensor<float> input(4, 1, 1, { 1.0, 2.0, 3.0, 4.0f });
    auto output = fc_layer(input);
    ASSERT_EQ(output->dims[0], 3);
    ASSERT_EQ(output->dims[1], 1);
    ASSERT_EQ(output->dims[2], 1);

    ASSERT_NEAR(8.48313684, output->data()[0], EPSILON);
    ASSERT_NEAR(-2.38187395, output->data()[1], EPSILON);
    ASSERT_NEAR(-1.31959815, output->data()[2], EPSILON);
}
