#include "cnncpp/network.hpp"
#include "cnncpp/layers/convolution.hpp"
#include "cnncpp/layers/pooling.hpp"
#include "cnncpp/layers/layers.hpp"
#include "cnncpp/tensor.hpp"
#include "highfive/H5File.hpp"

#include <algorithm>
#include <fstream>
#include <highfive/highfive.hpp>
#include <memory>

#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

static const std::unordered_map<std::string, cnncpp::activations::activation_func_ptr> _ACTICATIONS = { { "tanh", cnncpp::activations::tanh },
    { "relu", cnncpp::activations::relu } , {"softmax", cnncpp::activations::softmax}};

auto conv_layer_factory = [i = 0](const json& net_json, const HighFive::File& hd5_file) mutable -> std::unique_ptr<cnncpp::layer> {
    const auto& net_config = net_json["config"];
    std::string net_name("conv2d");
    if (i >= 1) {
        net_name += ("_" + std::to_string(i));
    }
    i++;
    auto input_shape = net_json["build_config"]["input_shape"];

    std::cout << input_shape << "\n";
    std::cout << input_shape.is_array() << "\n";
    std::cout << input_shape.size() << "\n";
    std::array<int, 3> input_shape_arr;
    std::copy_if(input_shape.begin(), input_shape.end(), input_shape_arr.begin(), [](auto val) { return !val.is_null(); });

    auto filters = net_config.value<int>("filters", -1);
    cnncpp::kernel_size_t kernel_size[2] { net_config["kernel_size"].at(0), net_config["kernel_size"].at(1) };

    cnncpp::stride_t strides[2] = { net_config["strides"].at(0), net_config["strides"].at(1) };
    auto padding = net_config["padding"];
    std::string activation = net_config["activation"];
    // read weights
    auto v0 = hd5_file.getDataSet("layers/" + net_name + "/vars/0");
    size_t total_size = 1;
    for (auto&& dim : v0.getDimensions()) {
        total_size *= dim;
    }
    std::vector<float> data(total_size);
    v0.read(data.data());

    auto v1 = hd5_file.getDataSet("layers/" + net_name + "/vars/1");
    total_size = 1;
    for (auto&& dim : v1.getDimensions()) {
        total_size *= dim;
    }
    std::vector<float> biases(total_size);
    v1.read(biases.data());

    return std::unique_ptr<cnncpp::layer>(new cnncpp::convolution(input_shape_arr, kernel_size[0], strides[0], filters, _ACTICATIONS.at(activation), data, biases));
};


auto fully_connected_layer_factory = [i = 0](const json& net_json, const HighFive::File& hd5_file) mutable -> std::unique_ptr<cnncpp::layer> {
    const auto& net_config = net_json["config"];
    std::string net_name("dense");
    if (i >= 1) {
        net_name += ("_" + std::to_string(i));
    }
    i++;
    auto output_shape =  net_config["units"];
    auto input_shape = net_json["build_config"]["input_shape"][1];

    std::cout << output_shape << "\n";
    std::cout << input_shape << "\n";
    
    
    std::string activation = net_config["activation"];
    // read weights
    auto v0 = hd5_file.getDataSet("layers/" + net_name + "/vars/0");
    size_t total_size = 1;
    for (auto&& dim : v0.getDimensions()) {
        total_size *= dim;
    }
    std::vector<float> data(total_size);
    v0.read(data.data());

    auto v1 = hd5_file.getDataSet("layers/" + net_name + "/vars/1");
    total_size = 1;
    for (auto&& dim : v1.getDimensions()) {
        total_size *= dim;
    }
    std::vector<float> biases(total_size);
    v1.read(biases.data());

    return std::unique_ptr<cnncpp::layer>(new cnncpp::fully_connected(input_shape, output_shape,  _ACTICATIONS.at(activation), data, biases));
};


auto avg_pool_layer_factory = [](const json& net_json, const HighFive::File& hd5_file) -> std::unique_ptr<cnncpp::layer> {

    const auto& net_config = net_json["config"];
    std::string net_name("avgpooling2d");

    auto input_shape = net_json["build_config"]["input_shape"];
    std::array<int, 3> input_shape_arr;
    std::copy_if(input_shape.begin(), input_shape.end(), input_shape_arr.begin(), [](auto val) { return !val.is_null(); });

    cnncpp::kernel_size_t kernel_size[2] { net_config["pool_size"].at(0), net_config["pool_size"].at(1) };

    cnncpp::stride_t strides[2] = { net_config["strides"].at(0), net_config["strides"].at(1) };
    auto padding = net_config["padding"];

    return std::unique_ptr<cnncpp::layer>(new cnncpp::avg_pool(input_shape_arr, kernel_size[0], strides[0]));
};

auto flatten_layer_factory = [](const json& net_json, const HighFive::File& hd5_file) -> std::unique_ptr<cnncpp::layer> {

    const auto& net_config = net_json["config"];
    auto input_shape = net_json["build_config"]["input_shape"];
    std::array<int, 3> input_shape_arr;
    std::copy_if(input_shape.begin(), input_shape.end(), input_shape_arr.begin(), [](auto val) { return !val.is_null(); });

    return std::unique_ptr<cnncpp::layer>(new cnncpp::flatten(input_shape_arr));
};


cnncpp::network::network(const std::string& json_config_file_path,
    const std::string& weights_hd5)
{
    auto model_config_json = json::parse(std::fstream(json_config_file_path));
    HighFive::File file(weights_hd5, HighFive::File::ReadOnly);
    for (auto& layer_config : model_config_json["config"]["layers"]) {
        std::cout << layer_config["class_name"] << " : " << layer_config["config"]["name"] << "\n";
        if (layer_config["class_name"] == "Conv2D") {
            _layers.push_back(conv_layer_factory(layer_config, file));
            continue;
        }
        if (layer_config["class_name"] == "AveragePooling2D") {
            _layers.push_back(avg_pool_layer_factory(layer_config, file));
            continue;
        }
        if (layer_config["class_name"] == "Flatten") {
            _layers.push_back(flatten_layer_factory(layer_config, file));
            continue;
        }
        if (layer_config["class_name"] == "Dense") {
            _layers.push_back(fully_connected_layer_factory(layer_config, file));
            continue;
        }
    }
}
const cnncpp::Tensor<float>& cnncpp::network::operator()(const Tensor<float>& input)
{
    const Tensor<float> *prev = &input;
    
    for (auto &&layer: _layers) {
        prev = (*layer)(*prev);
    }
    return _layers.back()->output();
}
