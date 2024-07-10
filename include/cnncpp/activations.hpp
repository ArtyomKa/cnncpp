#ifndef __CNNCPP_ACTIVATIONS_HPP__
#define __CNNCPP_ACTIVATIONS_HPP__
#include <algorithm>
#include <cmath>
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

}

#endif
