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
