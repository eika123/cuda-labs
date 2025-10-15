#include <stdio.h>
#include <cuda.h>
#include <cstdlib>

__global__ void cuda_hello_printf_convert() {
        printf("Hello from GPU thread %d in block %d\n", threadIdx.x,
               blockIdx.x);
}

int main(int argc, char **argv) {
        if (argc < 3) {
                fprintf(stderr,
                        "Usage: %s <number-of-blocks> <number-of-threads per "
                        "block>\n",
                        argv[0]);
                exit(1);
        }

        char *endptrt, *endptrb;
        // get number of blocks and threads per block from command line
        long number_of_blocks = strtol(argv[1], &endptrb, 10);
        long threads_per_block = strtol(argv[2], &endptrt, 10);

        // Launch the kernel on the GPU
        // <<<blockNum, threadsPerBlock>>> specifies the execution configuration:
        // 1 block of threads, with 1 thread per block
        cuda_hello_printf_convert<<<number_of_blocks, threads_per_block>>>();

        // Synchronize the device to ensure the kernel completes before the host
        // continues
        cudaDeviceSynchronize();

        printf("hello from host!");
}
