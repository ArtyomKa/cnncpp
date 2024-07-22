#include "cnncpp/activations.hpp"
#include "cnncpp/layers/convolution.hpp"
#include "cnncpp/layers/layers.hpp"
#include "cnncpp/layers/pooling.hpp"
#include "cnncpp/utils.hpp"
#include <algorithm>
#include <highfive/H5File.hpp>
#include <highfive/highfive.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, const char* argv[])
{
    if (argc != 3) {
        std::cout << "Incorrect number of arguments.\nUsage: cnncpp_runner <model_path> <image_path>";
        return -1;
    }
    std::string model_path(argv[1]);
    std::string image_path(argv[2]);
    auto image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    cv::resize(image, image, cv::Size(32, 32));
    std::cout << "image dims " << image.cols << "x" << image.rows << "x" << image.channels() << "\n";
    cv::Mat float_mat;
    image.convertTo(float_mat, CV_32F, 1.0 / 255.0);

    auto tensor = cnncpp::convert(float_mat);

    HighFive::File file(model_path, HighFive::File::ReadOnly);

    auto conv1_weights_data_set = file.getDataSet("/layers/conv2d_1/weights/kernels");
    auto conv1_bias_data_set = file.getDataSet("/layers/conv2d_1/weights/bias");
    auto conv1_weights = conv1_weights_data_set.read<std::vector<float>>();
    auto conv1_bias = conv1_bias_data_set.read<std::vector<float>>();

    auto conv2_weights_data_set = file.getDataSet("/layers/conv2d_2/weights/kernels");
    auto conv2_bias_data_set = file.getDataSet("/layers/conv2d_2/weights/bias");
    auto conv2_weights = conv2_weights_data_set.read<std::vector<float>>();
    auto conv2_bias = conv2_bias_data_set.read<std::vector<float>>();

    auto fc1_weights_data_set = file.getDataSet("/layers/fc_1/weights/kernels");
    auto fc1_bias_data_set = file.getDataSet("/layers/fc_1/weights/bias");
    auto fc1_weights = fc1_weights_data_set.read<std::vector<float>>();
    auto fc1_bias = fc1_bias_data_set.read<std::vector<float>>();

    auto fc2_weights_data_set = file.getDataSet("/layers/fc_2/weights/kernels");
    auto fc2_bias_data_set = file.getDataSet("/layers/fc_2/weights/bias");
    auto fc2_weights = fc2_weights_data_set.read<std::vector<float>>();
    auto fc2_bias = fc2_bias_data_set.read<std::vector<float>>();

    auto fc3_weights_data_set = file.getDataSet("/layers/fc_3/weights/kernels");
    auto fc3_bias_data_set = file.getDataSet("/layers/fc_3/weights/bias");
    auto fc3_weights = fc3_weights_data_set.read<std::vector<float>>();
    auto fc3_bias = fc3_bias_data_set.read<std::vector<float>>();

    auto conv2d_1 = cnncpp::convolution({ 32, 32, 1 }, 5, 1, 6, cnncpp::activations::tanh,
        conv1_weights, conv1_bias);
    auto avg_pool = cnncpp::avg_pool({ 28, 28, 6 }, 2, 2);
    auto conv2d_2 = cnncpp::convolution({ 14, 14, 6 }, 5, 1, 16, cnncpp::activations::tanh,
        conv2_weights, conv2_bias);
    auto avg_pool_2 = cnncpp::avg_pool({ 10, 10, 16 }, 2, 2);
    auto flatten = cnncpp::flatten({ 5, 5, 16 });
    auto fc1 = cnncpp::fully_connected(400, 120, cnncpp::activations::tanh, fc1_weights, fc1_bias);
    auto fc2 = cnncpp::fully_connected(120, 84, cnncpp::activations::tanh, fc2_weights, fc2_bias);
    auto fc3 = cnncpp::fully_connected(84, 10, cnncpp::activations::none, fc3_weights, fc3_bias);

    auto softmax = cnncpp::activations::softmax;
    auto output1 = conv2d_1(*tensor);
    auto output2 = avg_pool(*output1);
    auto output3 = conv2d_2(*output2);
    auto output4 = avg_pool_2(*output3);
    auto output5 = flatten(*output4);
    auto output6 = fc1(*output5);
    auto output7 = fc2(*output6);
    auto output8 = fc3(*output7);
    auto res = softmax(std::vector<float>(&output8->data()[0], &output8->data()[0] + output8->total()));

    std::cout << "Digit: " << std::max_element(res.begin(), res.end()) - res.begin() << std::endl;
    return 0;
}
