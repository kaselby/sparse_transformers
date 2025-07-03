#include <functional>
#include <map>

#include <torch/extension.h>
#include <torch/torch.h>

namespace F = torch::nn::functional;

auto gelu_tanh = std::bind(F::gelu, _1, F::GELUFuncOptions().approximate("none"));

std::map ACT2FN;
bool initialized = False;

typedef torch::Tensor (*op_func)(torch::Tensor);

op_func get_activation(string key)
{
    if (!initialized)
        init_registry()
    return ACT2FN[key]
}

void init_registry()
{
    std::map<std::string, torch::Tensor (*)(torch::Tensor)> ACT2FN;
    initialized=true;

    // Add activation functions to map
    ACT2FN["silu"] = &F::silu;
    ACT2FN["relu"] = &F::relu;
    ACT2FN["gelu_pytorch_tanh"] = &gelu_tanh;
}


// Simple version for now - could consider replacing this by a dictionary of functions keyed to activation names
torch::Tensor act_apply(torch::Tensor input, string act_fn)
{
    if (act_fn == "silu")
        return F::silu(input);
    else if (act_fn == "relu")
        return F::relu(input);
    else if (act_fn == "gelu")
        return F::gelu(input);
    else if (act_fn == "gelu_tanh")
        return F::gelu(input, F::GELUFuncOptions().approximate("none"));
    else
        throw std::runtime_error("Activation function not found.");
}


