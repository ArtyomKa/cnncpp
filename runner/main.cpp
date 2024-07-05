#include "utils.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[])
{
    auto image = cv::imread("./data/testSample/img_1.jpg");
    // cv::Mat blue = cv::Mat::zeros(20, 20, CV_8UC1);
    // cv::Mat green = cv::Mat::zeros(20, 20, CV_8UC1);
    cv::Mat red = cv::Mat::ones(20, 20, CV_8UC1) * 255;
    // std::vector<cv::Mat> channels { blue, green, red };
    // cv::Mat mat1(20, 20, CV_8UC(3));
    // cv::merge(&channels[0], 3, mat1);
    auto tensor = cnncpp::convert(image);
    //
    auto image2 = cnncpp::convert(*tensor.get());
    cv::imshow("Image-orig", image);
    cv::imshow("Image-after", image2);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
