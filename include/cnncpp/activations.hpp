#ifndef __CNNCPP_ACTIVATIONS_HPP__
#define __CNNCPP_ACTIVATIONS_HPP__
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
namespace cnncpp::activations {

typedef float (*activation_func_ptr)(float);

constexpr float relu(float val)
{
    return std::max(0.0f, val);
}
constexpr float tanh(float val)
{
    auto ex = exp(val);
    auto eminx = exp(-val);
    return (ex - eminx) / (ex + eminx);
}
constexpr float none(float val) { return val; }

inline std::vector<float> softmax(const std::vector<float>& input)
{
    std::vector<float> res(input.size());
    std::transform(input.begin(), input.end(), res.begin(), [](float v) { return exp(v); });
    auto sum = std::accumulate(res.begin(), res.end(), 0.0);

    std::transform(res.begin(), res.end(), res.begin(), [sum](float v) { return v / sum; });
    return res;
}
}

#endif
