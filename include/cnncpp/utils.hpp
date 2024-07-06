#ifndef __CNNCPP_UTILS_HPP__
#define __CNNCPP_UTILS_HPP__

#include "tensor.hpp"
#include <memory>
#include <opencv2/opencv.hpp>

namespace cnncpp {
std::unique_ptr<Tensor<float>> convert(cv::Mat image);
cv::Mat convert(const Tensor<float>& tensor);
}

#endif
