// hello_cuda.cu
#include <iostream>

// Kernel function executed on the GPU
__global__ void helloFromGPU() {
    printf("Hello World from GPU! Thread ID: %d\n", threadIdx.x);
}

int main() {
    // Launch kernel with 1 block and 5 threads
    helloFromGPU<<<1, 5>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    std::cout << "Hello World from CPU!" << std::endl;

    return 0;
}
