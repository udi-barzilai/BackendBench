from typing import Callable


__all__ = 'cuda_busy_wait',


# a simple kernel that loops for a target duration
cuda_src = """
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void busy_wait_kernel(long long target_clocks) {
    long long start = clock64();
    while (clock64() - start < target_clocks);
}

torch::Tensor busy_wait(long long target_clocks) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    busy_wait_kernel<<<1, 1, 0, stream>>>(target_clocks);
    return torch::empty({0});
}
"""

cpp_src = """
torch::Tensor busy_wait(long long target_clocks);
"""

# Load and launch kernel
from torch.utils.cpp_extension import load_inline
module = load_inline(
    name='delay',
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=['busy_wait'],
    with_cuda=True,
)
cuda_busy_wait: Callable[[int], None] = module.busy_wait
