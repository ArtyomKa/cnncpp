#include "cnncpp/utils.hpp"
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[])
{
    auto image = cv::imread("./data/testSample/img_1.jpg");
    std::cout << "image dims " << image.cols << "x" << image.rows << "x" << image.channels() << "\n";
    auto tensor = cnncpp::convert(image);
    //
    auto image2 = cnncpp::convert(*tensor.get());
    cv::imshow("Image-orig", image);
    cv::imshow("Image-after", image2);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
