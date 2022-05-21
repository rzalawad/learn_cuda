
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>


// kernel code
template <typename T>
__global__ void matmul(T *c, const T *a, const T *b, int N) {
    int x = (blockDim.x * blockIdx.x) + threadIdx.x;
    int y = (blockDim.y * blockIdx.y) + threadIdx.y;

    c[y*N+x] = 0;
    for (int i=0; i < N; ++i) {
        c[y*N+x] += a[y*N+i] + b[i*N+x];
    }
}

int main() {

    int N = 1 << 10; // 1024
    int M = 1 << 8;  // 256

    // Size (in bytes) of matrix
    size_t input_bytes = N * M * sizeof(int);
    size_t output_bytes = N * N * sizeof(int);

    // Host vectors
    std::vector<int> h_a(N * M);  // 1024 x 256
    std::vector<int> h_b(M * N);  // 256  x 1024
    std::vector<int> h_c(N * N);  // 1024 x 1024

    std::generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
    std::generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, input_bytes);
    cudaMalloc(&d_b, input_bytes);
    cudaMalloc(&d_c, output_bytes);

    cudaMemcpy(d_a, h_a.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), input_bytes, cudaMemcpyHostToDevice);

    int THREADS = 32;
    int BLOCKS = N / THREADS;  // NOTE: can cause problems when N is odd.

    std::cout << "Using block size of " << THREADS << "," << THREADS << std::endl;
    std::cout << "Using block dim of " << BLOCKS << "," << BLOCKS << std::endl;

    dim3 threads(THREADS, THREADS); // 32x32 = 1024
    dim3 blocks(BLOCKS, BLOCKS);    // 32x32 = 1024

    matmul <<<blocks, threads>>> (d_c, d_a, d_b, N);

    cudaMemcpy(h_c.data(), d_c, output_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
