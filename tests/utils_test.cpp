#include "cnncpp/utils.hpp"
#include "gtest/gtest.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

// TEST(UtilsTest, TestOpencvConvert)
// {
//     cv::Mat original;
//     // create a blue 20x20 "image"
//     cv::merge(std::vector<cv::Mat> { cv::Mat::zeros(20, 20, CV_8UC1),
//                   cv::Mat::zeros(20, 20, CV_8UC1),
//                   cv::Mat::ones(20, 20, CV_8UC1) * 255 },
//         original);
//     auto tensor = cnncpp::convert(original);
//     //
//     auto converted = cnncpp::convert(*tensor.get());
//     ASSERT_TRUE(std::equal(&converted.data[0], &converted.data[0] + converted.total(), &original.data[0]));
// }
