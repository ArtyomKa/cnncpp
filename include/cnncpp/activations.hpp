#ifndef __CNNCPP_ACTIVATIONS_HPP__
#define __CNNCPP_ACTIVATIONS_HPP__
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
namespace cnncpp::activations {

inline void softmax(std::vector<float>& input)
{
    float sum = 0.0;
    std::transform(input.begin(), input.end(), input.begin(), [&sum](float v) {
        auto exp_v = exp(v);
        sum += exp_v;
        return exp_v;
    });

    std::transform(input.begin(), input.end(), input.begin(), [sum](float v) { return v / sum; });
}

inline void relu(std::vector<float>& data)
{
    std::transform(data.cbegin(), data.cend(), data.begin(), [](float val) {
        return std::max(0.0f, val);
    });
}

inline void tanh(std::vector<float>& data)
{
    std::transform(data.cbegin(), data.cend(), data.begin(), [](float val) {
        auto ex = exp(val);
        auto eminx = exp(-val);
        return (ex - eminx) / (ex + eminx);
    });
}

inline void none(std::vector<float>& data) { return; };

using activation_func_ptr = void (*)(std::vector<float>&);
}

#endif
