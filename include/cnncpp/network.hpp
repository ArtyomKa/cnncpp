#ifndef __CNNCPP_NETWORK_HPP__
#define __CNNCPP_NETWORK_HPP__
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "cnncpp/layers/layers.hpp"
namespace cnncpp {

class network {
    std::vector<std::unique_ptr<layer>> _layers;

public:
    explicit network(const std::string& json_config_file_path,
        const std::string& weights_hd5);

    const Tensor<float>& operator()(const Tensor<float>& input);
};

}
#endif
