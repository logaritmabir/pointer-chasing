#include "pch.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <ctime>
#include <cstdint>
#include <fstream>
#include "../pointer-chasing/kernel.cu"

__global__ void dram_latency(int* data, int* dram_delay);
__global__ void register_latency(int* data, int* reg_delay);
__global__ void initializeData(int* data);

TEST(CudaKernelsTest, InitializeData) {
    const int N = 1024 * 32;
    int* data;
    cudaMallocManaged(&data, N * sizeof(int));

    dim3 block(1024);
    dim3 grid(N / 1024);
    initializeData<<<grid, block>>>(data);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++) {
        EXPECT_EQ(data[i], i % 1024);
    }

    cudaFree(data);
}

TEST(CudaKernelsTest, DramLatency) {
    const int N = 1024 * 32;
    int* data, * dram_delay;
    cudaMallocManaged(&data, N * sizeof(int));
    cudaMallocManaged(&dram_delay, 1024 * sizeof(int));

    dim3 block(1024);
    dim3 grid(N / 1024);
    initializeData<<<grid, block>>>(data);
    cudaDeviceSynchronize();
    dram_latency<<<1, 1024>>>(data, dram_delay);
    cudaDeviceSynchronize();

    for (int i = 0; i < 1024; i++) {
        EXPECT_GE(dram_delay[i], 0);
    }

    cudaFree(data);
    cudaFree(dram_delay);
}

TEST(CudaKernelsTest, RegisterLatency) {
    const int N = 1024 * 32;
    int* data, * reg_delay;
    cudaMallocManaged(&data, N * sizeof(int));
    cudaMallocManaged(&reg_delay, 1024 * sizeof(int));

    register_latency<<<1, 1024>>>(data, reg_delay);
    cudaDeviceSynchronize();

    for (int i = 0; i < 1024; i++) {
        EXPECT_GE(reg_delay[i], 0);
    }

    cudaFree(data);
    cudaFree(reg_delay);
}

// Ek testler
TEST(CudaKernelsTest, DramLatencyWithDifferentData) {
    const int N = 1024 * 32;
    int* data, * dram_delay;
    cudaMallocManaged(&data, N * sizeof(int));
    cudaMallocManaged(&dram_delay, 1024 * sizeof(int));

    // Farklý veri ile initialize
    for (int i = 0; i < N; i++) {
        data[i] = N - i;
    }

    dram_latency<<<1, 1024>>>(data, dram_delay);
    cudaDeviceSynchronize();

    for (int i = 0; i < 1024; i++) {
        EXPECT_GE(dram_delay[i], 0);
    }

    cudaFree(data);
    cudaFree(dram_delay);
}

TEST(CudaKernelsTest, RegisterLatencyWithDifferentData) {
    const int N = 1024 * 32;
    int* data, * reg_delay;
    cudaMallocManaged(&data, N * sizeof(int));
    cudaMallocManaged(&reg_delay, 1024 * sizeof(int));

    // Farklý veri ile initialize
    for (int i = 0; i < N; i++) {
        data[i] = N - i;
    }

    register_latency<<<1, 1024>>>(data, reg_delay);
    cudaDeviceSynchronize();

    for (int i = 0; i < 1024; i++) {
        EXPECT_GE(reg_delay[i], 0);
    }

    cudaFree(data);
    cudaFree(reg_delay);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
