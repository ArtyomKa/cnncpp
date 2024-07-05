#include "utils.hpp"
#include "gtest/gtest.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

TEST(UtilsTest, TestOpencvConvert)
{
    auto image = cv::imread("../data/testSample/img_1.jpg");
    auto tensor = cnncpp::convert(image);

    auto image2 = cnncpp::convert(*tensor.get());
    cv::imshow("Image", image2);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
