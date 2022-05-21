
#include <iostream>
#define SIZE 100000


// kernel code
// [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
// [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
// [ 2,  4,  6,  8, 10, 12, 14, 16, 18, 20]
// blockDim = 5
template <typename T>
__global__ void add(T *c, const T *a, const T *b) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

int main() {
    int a[SIZE] = {1};
    int b[SIZE] = {2};
    int out[SIZE] = {0};

    size_t size = SIZE * sizeof(int);
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    std::cout << "Hello World\n";
    std::cout << a[0] << "\n";
    std::cout << out[0] << "\n";

    add<<<200, 500>>>(d_c, d_a, d_b);
    cudaMemcpy(out, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << out[0] << "\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
