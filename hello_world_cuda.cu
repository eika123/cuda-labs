#include <stdio.h>

__global__ void cuda_hello_printf_convert() {
    printf("Hello from GPU\n");
}

int main() { 


    // Launch the kernel on the GPU
    // <<<1, 1>>> specifies the execution configuration:
    // 1 block of threads, with 1 thread per block
    cuda_hello_printf_convert<<<1, 1>>>();

    // Synchronize the device to ensure the kernel completes before the host continues
    cudaDeviceSynchronize(); 

    printf("hello from host!");
}
