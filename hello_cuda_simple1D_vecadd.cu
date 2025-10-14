#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <time.h>

#include <cstring>
#include <ctime>

#define MAX(x, y) ((x) > (y)) ? (x) : (y)


/**
 * Compare times for vector addition on GPU and CPU.
 * None of the codes are optimized in any way, solutions are as simple as possible
 */

// Device code: call with one-dimensional thread block
__global__ void VecAdd(float* A, float* B, float* C, int N) {
        int i = threadIdx.x;
        if (i < N) C[i] = A[i] + B[i];
}

__global__ void SetVal(float* A, float value, int N) {
        int i = threadIdx.x;
        A[i] = value;
}

void set_value_on_host(float* A, float value, int N) {
        int i;
        for (i = 0; i < N; i++) {
                A[i] = value;
        }
}

void vec_add_host(const float* A, const float* B, float* C, int N) {
        int i;
        for (i = 0; i < N; i++) {
                C[i] = A[i] + B[i];
        }
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

void print_vector_add_rows(float* A, float* B, float* C, int N) {
        printf("vector A:  ");
        print_vector_row(A, N);

        printf("vector B:  ");
        print_vector_row(B, N);

        printf("vector C:  ");
        print_vector_row(C, N);
}

void print_elapsed_time_seconds(timespec* start, timespec* end) {
        float nanosecs_in_secf = 1e9;
        int nanosecs_in_sec_int = (int)nanosecs_in_secf;
        time_t tv_sec_start = start->tv_sec;
        long tv_nanosec_start = start->tv_nsec;

        time_t tv_sec_end = end->tv_sec;
        long tv_nanosec_end = end->tv_nsec;

        time_t seconds_elapsed =
            MAX(((long)tv_sec_end - (long)tv_sec_start - 1), 0);
        long nanoseconds_elapsed =
            MAX((tv_nanosec_end + (nanosecs_in_sec_int - tv_nanosec_start)), 0);

        long extrasec = nanoseconds_elapsed / nanosecs_in_sec_int;
        if (extrasec) {
                seconds_elapsed += 1;
                nanoseconds_elapsed -= nanosecs_in_sec_int;
        }

        double secs_elapsed =
            (double)seconds_elapsed +
            ((double)nanoseconds_elapsed) / (nanosecs_in_secf);

        printf("seconds elapsed: %3.10f\n", secs_elapsed);
}

// Host code
int main(int argc, char** argv) {
        if (argc < 2) {
                printf("usage: %s <N>\nWhere N is the length of the vectors",
                       argv[0]);
                exit(1);
        }

        timespec ts_cuda_start, ts_cuda_end, ts_host_start, ts_host_end;
        int N = atoi(argv[1]);
        size_t size = N * sizeof(float);
        float* h_A;
        if ((h_A = (float*)calloc(N, sizeof(float))) == NULL) {
                printf("memory allocation failed\n");
                exit(1);
        }
        float* h_B;
        if ((h_B = (float*)calloc(N, sizeof(float))) == NULL) {
                printf("memory allocation failed\n");
                exit(1);
        }
        float* h_C;
        if ((h_C = (float*)calloc(N, sizeof(float))) == NULL) {
                printf("memory allocation failed\n");
                exit(1);
        }

        printf("\n =========== GPU addition =================== \n");
        if (clock_gettime(CLOCK_REALTIME, &ts_cuda_start) == -1) {
                printf("getting clock startttime failed");
        }
        // device allocations
        float* d_A;
        float* d_B;
        float* d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        SetVal<<<1, N>>>(d_A, 1.0, N);
        SetVal<<<1, N>>>(d_B, 2.0, N);
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        SetVal<<<1, N>>>(d_A, 1.0, N);
        SetVal<<<1, N>>>(d_B, 2.0, N);

        VecAdd<<<1, N>>>(d_A, d_B, d_C, N);

        cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        if (clock_gettime(CLOCK_REALTIME, &ts_cuda_end) == -1) {
                printf("getting clock end time failed");
        }
        printf("elapsed time for GPU addition:\n");
        print_elapsed_time_seconds(&ts_cuda_start, &ts_cuda_end);
        printf("\n =========== host addition =================== \n");

        // print_vector_add_rows(h_A, h_B, h_C, N);
        set_value_on_host(h_C, 0.0, N);

        if (clock_gettime(CLOCK_REALTIME, &ts_host_start) == -1) {
                printf("getting clock startttime failed");
        }
        set_value_on_host(h_A, 1.0, N);
        set_value_on_host(h_B, 2.0, N);

        vec_add_host(h_A, h_B, h_C, N);
        if (clock_gettime(CLOCK_REALTIME, &ts_host_end) == -1) {
                printf("getting clock end timefailed");
        }

        printf("elapsed time for host addition\n");
        print_elapsed_time_seconds(&ts_host_start, &ts_host_end);

        printf("host: h_C[9* N/10] = %f\n", h_C[9 * (N / 10)]);
}
