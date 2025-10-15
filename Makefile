

.PHONY: all clean


all: hello_cuda hello_vecadd_cuda_simple1D hello_vecadd_cuda_blocks hello_cuda_blocks

hello_cuda: 
	nvcc hello_world_cuda.cu -o $@

hello_vecadd_cuda_simple1D:
	nvcc hello_cuda_simple1D_vecadd.cu -o $@

hello_vecadd_cuda_blocks:
	nvcc hello_cuda_vecadd_blocks.cu -o $@

hello_cuda_blocks:
	nvcc hello_cuda_blocks.cu -o $@

clean:
	rm -f hello_cuda hello_vecadd_cuda_simple1D hello_vecadd_cuda_blocks hello_cuda_blocks
