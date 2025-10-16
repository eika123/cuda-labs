#include <cuda_runtime_api.h>
#include <stdio.h>
#include <cuda.h>

/**
 * Example from page 301 in 
 * Peter S. Pacheco & Matthew Malensek,
 * "An Introduction to Parallel Programming", 2nd edition. 
 * Morgan Kaufmann
 */

__global__ void cuda_hello_blocks() {
        printf(
            "Hello from GPU thread: "
            "block.x=%d, block.y=%d, block.z=%d threadIdx.x=%d threadIdx.y=%d "
            "threadIdx.z=%d\n",
            blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y,
            threadIdx.z);
}

int main(void) {

    dim3 grid_dims, block_dims;

    grid_dims.x = 2;
    grid_dims.y = 3;
    grid_dims.z = 1;

    block_dims.x=4;
    block_dims.y=4;
    block_dims.z=4;

    cuda_hello_blocks<<<grid_dims, block_dims>>>();
    cudaDeviceSynchronize();
}

