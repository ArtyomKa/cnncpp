#include "cnncpp/utils.hpp"
#include "cnncpp/tensor.hpp"
#include <algorithm>
#include <memory>
#include <opencv2/core/hal/interface.h>
#include <vector>

std::unique_ptr<cnncpp::Tensor<float>> cnncpp::convert(cv::Mat image)
{
    auto total = image.cols * image.rows * image.channels();
    std::vector<float> temp(total, 0);
    std::copy(image.begin<float>(), image.end<float>(), temp.begin());
    // std::transform(image.begin<float>(), image.end<float>(), temp.begin(), [](float val) { return val });
    return std::make_unique<Tensor<float>>(image.rows, image.cols, image.channels(), temp);
}

/*********************************************************************
 *  Assuming that the data in the tensor already in [0-255] range
 *********************************************************************/
cv::Mat cnncpp::convert(const cnncpp::Tensor<float>& tensor)
{

    cv::Mat res(tensor.dims[0], tensor.dims[1], CV_8UC(tensor.dims[2]));
    std::transform(tensor.data(), tensor.data() + tensor.total(), &res.data[0], [](float val) { return static_cast<uchar>(val); });
    return res;
}
