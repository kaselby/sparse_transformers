/*
// For TorchScript support
#include <torch/script.h>

// For PyTorch C++ extension support
#include <torch/extension.h>

// For tensor operations
#include <ATen/ATen.h>

// For PyTorch's OpenMP wrapper
#include <ATen/ParallelOpenMP.h>

// Add pybind11 and namespace
#include <pybind11/pybind11.h>
namespace py = pybind11;

// Add required headers
#include <future>
#include <thread>
#include <mutex>

// Add device check utilities
#include <c10/cuda/CUDAGuard.h>

// Add custom headers
#include "weight_cache_opt.h"
#include "approx_topk.h"
*/

// Forward declarations of CPU/CUDA implementations
torch::Tensor sparse_mlp_forward_cpu_opt(
    const torch::Tensor &input,
    const torch::Tensor &active_up_weight,
    const torch::Tensor &active_down_weight,
    torch::Tensor &down_proj_buffer,
    torch::Tensor &up_proj_buffer,
    const std::string &activation_fn);

#ifdef WITH_CUDA
torch::Tensor sparse_mlp_forward_cuda_opt(
    const torch::Tensor &input,
    const torch::Tensor &active_up_weight,
    const torch::Tensor &active_down_weight,
    torch::Tensor &down_proj_buffer,
    torch::Tensor &up_proj_buffer,
    const std::string &activation_fn);
#endif

// Main dispatch function
torch::Tensor sparse_mlp_forward_opt(
    const torch::Tensor &input,
    const torch::Tensor &active_up_weight,
    const torch::Tensor &active_down_weight,
    torch::Tensor &down_proj_buffer,
    torch::Tensor &up_proj_buffer,
    const std::string &activation_fn)
{

    // Check if input is on CUDA and dispatch accordingly
    if (input.is_cuda())
    {
#ifdef WITH_CUDA
        return sparse_mlp_forward_cuda_opt(input, active_up_weight, active_down_weight, down_proj_buffer, up_proj_buffer, activation_fn);
#else
        AT_ERROR("CUDA not available - cannot run on GPU");
#endif
    }
    else
    {
        return sparse_mlp_forward_cpu_opt(input, active_up_weight, active_down_weight, down_proj_buffer, up_proj_buffer, activation_fn);
    }
}

// CPU implementation
torch::Tensor sparse_mlp_forward_cpu_opt(
    const torch::Tensor &input,
    const torch::Tensor &active_up_weight,
    const torch::Tensor &active_down_weight,
    torch::Tensor &down_proj_buffer,
    torch::Tensor &up_proj_buffer,
    const std::string &activation_fn)
{
    // Store original input shape for reshaping output later
    auto original_shape = input.sizes().vec();
    bool needs_reshape = input.dim() > 2;

    // Flatten input if it has more than 2 dimensions
    torch::Tensor input_2d;
    if (needs_reshape)
    {
        // Flatten all dimensions except the last one (hidden dimension)
        auto hidden_size = original_shape.back();
        auto total_batch_size = input.numel() / hidden_size;
        input_2d = input.view({total_batch_size, hidden_size});
    } else {
        input_2d = input;
    }

    const auto batch_size = input_2d.size(0);
    const auto hidden_size = input_2d.size(1);

    // Ensure output buffer is correctly sized
    // Check both dimensions to avoid resize warnings
    const int64_t gate_size = active_up_weight.size(0);

    // For down_proj_buffer: [batch_size, hidden_size]
    if (down_proj_buffer.size(0) != batch_size || down_proj_buffer.size(1) != hidden_size)
        down_proj_buffer.resize_({batch_size, hidden_size});

    // For combined_proj_buffer: [batch_size, 2 * gate_size]
    if (up_proj_buffer.size(0) != batch_size || up_proj_buffer.size(1) != gate_size)
        up_proj_buffer.resize_({batch_size, gate_size});

    // Optimal grain size for heavy matmul operations
    const int64_t num_threads = at::get_num_threads();
    int64_t grain_size = 1;
    
    if (batch_size > num_threads && num_threads > 0) {
        // Base calculation: create 2-4x more work chunks than threads
        const int64_t target_chunks = num_threads * 3;  // 3x oversubscription
        grain_size = std::max(int64_t(1), batch_size / target_chunks);
        
        // Cap at 32 for heavy operations (much smaller than 64 for light ops)
        const int64_t max_grain_matmul = 32;
        grain_size = std::min(grain_size, max_grain_matmul);
    }

    // Process each batch block in parallel
    at::parallel_for(0, batch_size, grain_size, [&](int64_t start, int64_t end)
                     {
        // Process blocks of batches instead of single items
        const int64_t block_size = end - start;
        //const int64_t gate_size = concat_weight.size(0) / 2;
        
        // Get input block for this thread
        auto input_block = input_2d.slice(0, start, end);  // [block_size, hidden_size]
        
        // Get output buffer views for this block
        auto up_proj_block = up_proj_buffer.slice(0, start, end);  // [block_size, gate_size]
        auto down_proj_block = down_proj_buffer.slice(0, start, end);  // [block_size, hidden_size]
        
        // Perform batch matrix multiplication for gate and up projections
        // This is more efficient than individual matmuls
        torch::matmul_out(up_proj_block, input_block, active_up_weight.t());
        
        // Split into gate and up projections
        //auto gate_proj = combined_proj_block.narrow(1, 0, gate_size);  // [block_size, gate_size]
        //auto up_proj = combined_proj_block.narrow(1, gate_size, gate_size);  // [block_size, gate_size]

        up_proj_block.sigmoid_();  // In-place sigmoid
        //gate_proj.mul_(up_proj);  // In-place element-wise multiplication
        
        // Final projection to output dimension
        torch::matmul_out(down_proj_block, up_proj_block, active_down_weight.t()); });

    // Reshape output back to original shape if input was multi-dimensional
    return needs_reshape ? down_proj_buffer.view(original_shape) : down_proj_buffer;
}