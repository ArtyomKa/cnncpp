
#include "cnncpp/network.hpp"
#include "cnncpp/utils.hpp"
#include "highfive/H5File.hpp"
#include <fstream>
#include <highfive/highfive.hpp>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
using json = nlohmann::json;
int main(int argc, const char* argv[])
{
    std::string model_file_name("/home/artyom/devel/sandbox/cnncpp/data/model/model.weights.h5");
    std::string model_config_file_name("/home/artyom/devel/sandbox/cnncpp/data/model/config.json");
    // if (argc != 3) {
        // std::cout << "Incorrect number of arguments.\nUsage: cnncpp_runner <model_path> <image_path>";
        // return -1;
    // }
    
    std::string image_path("/home/artyom/devel/sandbox/cnncpp/data/testSet/testSet/img_1.jpg");
    auto image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    cv::resize(image, image, cv::Size(32, 32));
    std::cout << "image dims " << image.cols << "x" << image.rows << "x" << image.channels() << "\n";
    cv::Mat float_mat;
    image.convertTo(float_mat, CV_32F, 1.0 / 255.0);

    auto tensor = cnncpp::convert(float_mat);
    cnncpp::network net(model_config_file_name, model_file_name);
    auto &output = net(*tensor);
    // auto res = cnncpp::activations::softmax(std::vector<float>(&output.data()[0], &output.data()[0] + output.total()));
    std::cout << "Done\n";
}
