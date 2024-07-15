
#include "highfive/H5File.hpp"
#include <highfive/highfive.hpp>

int main(int argc, const char* argv[])
{
    std::string model_file_name = "/home/artyom/devel/sandbox/cnncpp/data/model/model.weights.h5";

    HighFive::File file(model_file_name, HighFive::File::ReadOnly);
    auto names = file.listObjectNames();
    for (auto& name : names) {
        std::cout << name << "\n";
    }
    auto layers_obj = file.getGroup("layers/conv2d/vars/");

    for (auto& name : layers_obj.listObjectNames()) {
        std::cout << name << "\n";
    }
    size_t total_size = 1;
    auto v0 = file.getDataSet("layers/conv2d/vars/0");
    for (auto &dim : v0.getDimensions()) {
        total_size *= dim;
    }
    std::vector<float> data(total_size);
    v0.read_raw<float>(data.data());
    std::cout << "Done\n";
}
