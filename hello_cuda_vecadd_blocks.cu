#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>

#define MAX(x, y) ((x) > (y)) ? (x) : (y)
#define MIN(x, y) ((x) < (y)) ? (x) : (y)

__global__ void vec_add_cuda(const float *x, const float *y, float *z,
                             const int N) {
        // assume one-dimensional blocks and a one-dimensional block-grid
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N) z[idx] = x[idx] + y[idx];
}

__global__ void vec_set_val_cuda(float *x, float value, int N) {
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx < N) x[idx] = value;
}

int get_args(int argc, char **argv, int *N, long *blk_cnt, long *thr_per_blk) {
        if (argc < 2) {
                fprintf(stderr, "Usage: %s <vector-size>", argv[0]);
                return 0;
        }
        *N = atoi(argv[1]);
        if (*N <= 1) {
                fprintf(stderr, "N <= 1 is invalid\n");
        }

        *blk_cnt = (*N / 1024) + 1;
        *thr_per_blk = 1024;
        if (*blk_cnt == 1) {
                *thr_per_blk = MAX(MIN(*N, *thr_per_blk), 0);
        }
        return 1;
}

/**
 * Allocate three cuda float vectors x, y and z of length N.
 * If an allocation fails, the preceding allocations are cleaned up and a value
 * is returned.
 */
int allocate_vectors_cuda(float **x, float **y, float **z, int N) {
        cudaError_t cdError_x, cdError_y, cdError_z;

        cdError_x = cudaMalloc(x, N * sizeof(float));
        if (cdError_x) goto cleanup_x;

        cdError_y = cudaMalloc(y, N * sizeof(float));
        if (cdError_y) goto cleanup_y;

        cdError_z = cudaMalloc(z, N * sizeof(float));
        if (cdError_z) goto cleanup_z;

        return 1;

        // cleanup
cleanup_z:
        cudaFree(y);

cleanup_y:
        cudaFree(x);

cleanup_x:
        return 0;
}

/**
 * Allocate three float vectors x, y and z of length N on the host.
 * If an allocation fails, the preceding allocations are cleaned up and a value
 * is returned.
 */
int allocate_vectors_host(float **x, float **y, float **z, int N) {
        *x = (float *)calloc(N, sizeof(float));
        if (x == NULL) goto cleanup_x;

        *y = (float *)calloc(N, sizeof(float));
        if (y == NULL) goto cleanup_y;

        *z = (float *)calloc(N, sizeof(float));
        if (z == NULL) goto cleanup_z;

        return 1;

cleanup_z:
        free(*y);
cleanup_y:
        free(*x);
cleanup_x:
        return 0;
}

void copy_device_to_host(float *hst_x, float *hst_y, float *hst_z,
                         const float *dev_x, const float *dev_y,
                         const float *dev_z, int N) {
        cudaMemcpy(hst_x, dev_x, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hst_y, dev_y, N * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hst_z, dev_z, N * sizeof(float), cudaMemcpyDeviceToHost);
}

void print_vector_row(float arr[], int num_of_elements) {
        int i;

        printf("[ ");
        for (i = 0; i < num_of_elements; i++) {
                printf("%5.3f", arr[i]);
                if (i < num_of_elements - 1) {
                        printf(", ");
                }
        }
        printf(" ]\n");
}

void print_vector_add_rows(float *A, float *B, float *C, int N) {
        printf("vector A:  ");
        print_vector_row(A, N);

        printf("vector B:  ");
        print_vector_row(B, N);

        printf("vector C:  ");
        print_vector_row(C, N);
}

/**
 * Returns 0 on error, 1 on success
 */
int within_tolerance(float *A, float expected, float tol, int N,
                     const char *msg) {
        int j;
        for (j = 0; j < N; j++) {
                if (fabs(A[j] - expected) > tol) {
                        if (msg) {
                                printf("%s: ", msg);
                        }
                        printf(
                            "error in cuda vector assignment, encountered "
                            "value %f, expected value %f\n",
                            A[j], expected);
                        return 0;
                }
        }
        return 1;
}

int main(int argc, char **argv) {
        int N;
        long blk_cnt, thr_per_blk;
        int arg_success = get_args(argc, argv, &N, &blk_cnt, &thr_per_blk);
        printf("args received: N=%d\n", N);
        printf("args deduced: blk_cnt=%ld   thr_per_blk=%ld\n", blk_cnt,
               thr_per_blk);

        float *dev_x, *dev_y, *dev_z;
        float *hst_x, *hst_y, *hst_z;
        if (!allocate_vectors_cuda(&dev_x, &dev_y, &dev_z, N)) {
                fprintf(stderr, "cuda device memory allocation failed\n");
                exit(1);
        }
        if (!allocate_vectors_host(&hst_x, &hst_y, &hst_z, N)) {
                fprintf(stderr, "host memory allocation failed\n");
                exit(1);
        }

        vec_set_val_cuda<<<blk_cnt, thr_per_blk>>>(dev_x, 1.0f, N);
        vec_set_val_cuda<<<blk_cnt, thr_per_blk>>>(dev_y, 2.0f, N);
        vec_set_val_cuda<<<blk_cnt, thr_per_blk>>>(dev_z, 0.0f, N);
        vec_add_cuda<<<blk_cnt, thr_per_blk>>>(dev_x, dev_y, dev_z, N);

        cudaDeviceSynchronize();
        copy_device_to_host(hst_x, hst_y, hst_z, dev_x, dev_y, dev_z, N);

        if (N < 25) print_vector_add_rows(hst_x, hst_y, hst_z, N);

        int x, y, z;
        float tol = 1e-10;
        x = within_tolerance(hst_x, 1.0f, tol, N, "x");
        y = within_tolerance(hst_y, 2.0f, tol, N, "y");
        z = within_tolerance(hst_z, 3.0f, tol, N, "z");
        if (!(x && y && z)) {
                fprintf(stderr, "cuda vector addition failed\n");
                return 1;
        }

        return 0;
}
