
#include "ops.hpp"
#include "gtest/gtest.h"

TEST(KernelTest, CreationTest)
{

    auto kernel_data = { -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 };
    cnncpp::kernel kernel(3, 3, kernel_data);

    EXPECT_TRUE(std::equal(kernel.data.begin(), kernel.data.end(), kernel_data.begin()));
}
