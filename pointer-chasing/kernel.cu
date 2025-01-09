#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <ctime>
#include <cstdint>
#include <fstream>
#include <stdio.h>
#include <curand_kernel.h>

#define THREAD_PER_BLOCK 64
#define BLOCKS 16
#define TOTAL_THREADS (THREAD_PER_BLOCK * BLOCKS)
#define L1_CACHE_SIZE_IN_BYTES 64 * 1024
#define L2_CACHE_SIZE_IN_BYTES 1536 * 1024

#define CONSTANT_MEMORY_SIZE_IN_BYTES 32 * 1024
#define MAX_CONSTANT_MEMORY_ELEMENTS CONSTANT_MEMORY_SIZE_IN_BYTES / sizeof(int32_t)

#define ITERATIONS 1024

__device__ __constant__ int32_t d_const_data[MAX_CONSTANT_MEMORY_ELEMENTS];

__global__ void constant_memory_latency(int32_t* dummy, float* const_delay) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t value = 0;
	int32_t result = 0;
	clock_t start = clock();
	for (int i = 0; i < ITERATIONS; i++) {
		value = d_const_data[value];
		result += value;
	}
	clock_t end = clock();
	dummy[idx] = result;
	const_delay[idx] = static_cast<float>(end - start) / static_cast<float>(ITERATIONS);
}

__global__ void dram_latency_random(int32_t* data, int32_t* dummy, float* dram_delay, unsigned int seed, int32_t data_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed + idx, 0, 0, &state);
    int32_t start_idx = curand(&state) % data_length;

    int32_t result = 0;

    clock_t start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        start_idx = data[start_idx];
        result += start_idx;
    }
    clock_t end = clock();
    dummy[idx] = result;
    dram_delay[idx] = static_cast<float>(end - start) / static_cast<float>(ITERATIONS);
}

__global__ void dram_latency_broadcast(int32_t* data, int32_t* dummy, float* dram_delay, int32_t data_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t value = 0;
    int32_t result = 0;
    clock_t start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        value = data[value];
        result += value;
    }
    clock_t end = clock();
    dummy[idx] = result;
    dram_delay[idx] = static_cast<float>(end - start) / static_cast<float>(ITERATIONS);
}

__global__ void register_latency(int32_t* dummy, float* reg_delay) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t r, x = 2, y = 5, z = 7, q = 1, p = 3;

    clock_t start = clock();
    for (int i = 0; i < ITERATIONS; i++) {
        r = p; p = q; q = x; x = y; y = z; z = r;
    }
    clock_t end = clock();

    dummy[idx] = r;
    reg_delay[idx] = static_cast<float>(end - start) / static_cast<float>(ITERATIONS);
}

#define CHECK_CUDA_ERROR(err, msg) \
    if (err != cudaSuccess) { \
        std::cerr << msg << ": " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

void printDeviceProperties() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "Device Name: " << deviceProp.name << std::endl;
    std::cout << "L2 Cache Size: " << deviceProp.l2CacheSize << " bytes" << std::endl;
    std::cout << "Constant Memory Size: " << deviceProp.totalConstMem << " bytes" << std::endl;
    std::cout << "Shared Memory Size per Block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "Global Memory Size: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
}


int main() {
	printDeviceProperties();
    const int32_t data_length = 256 * 1024 * 1024; // 1GB (uint32_t)
    int32_t* data, *const_data, * dummy;
    float* dram_delay_random, *dram_delay_broadcast, * reg_delay, *const_delay;

    CHECK_CUDA_ERROR(cudaMallocManaged(&data, data_length * sizeof(int32_t)), "Failed to allocate data");
    CHECK_CUDA_ERROR(cudaMallocManaged(&dummy, TOTAL_THREADS * sizeof(int32_t)), "Failed to allocate dummy");
    CHECK_CUDA_ERROR(cudaMallocManaged(&dram_delay_random, TOTAL_THREADS * sizeof(float)), "Failed to allocate dram_delay");
    CHECK_CUDA_ERROR(cudaMallocManaged(&dram_delay_broadcast, TOTAL_THREADS * sizeof(float)), "Failed to allocate dram_delay_broadcast");
    CHECK_CUDA_ERROR(cudaMallocManaged(&reg_delay, TOTAL_THREADS * sizeof(float)), "Failed to allocate reg_delay");
    CHECK_CUDA_ERROR(cudaMallocManaged(&const_delay, TOTAL_THREADS * sizeof(float)), "Failed to allocate const_delay");

    for (int32_t i = 0; i < data_length; i++) {
        data[i] = rand() % data_length;
    }

	const_data = new int32_t[MAX_CONSTANT_MEMORY_ELEMENTS];
    for (int32_t i = 0; i < MAX_CONSTANT_MEMORY_ELEMENTS; i++) {
        const_data[i] = rand() % MAX_CONSTANT_MEMORY_ELEMENTS;
    }

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_const_data, const_data, MAX_CONSTANT_MEMORY_ELEMENTS * sizeof(uint32_t)), "Failed to copy const_data to device");

	constant_memory_latency << <BLOCKS, THREAD_PER_BLOCK >> > (dummy, const_delay);
    CHECK_CUDA_ERROR(cudaGetLastError(), "constant_memory_latency launch failed");
    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "constant_memory_latency execution failed");

    dram_latency_broadcast<<<BLOCKS, THREAD_PER_BLOCK>>>(data, dummy, dram_delay_broadcast, data_length);
    CHECK_CUDA_ERROR(cudaGetLastError(), "dram_latency_broadcast launch failed");
    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "dram_latency_broadcast execution failed");

    unsigned int seed = static_cast<unsigned int>(time(NULL));
    dram_latency_random<<<BLOCKS, THREAD_PER_BLOCK>>>(data, dummy, dram_delay_random, seed, data_length);
    CHECK_CUDA_ERROR(cudaGetLastError(), "dram_latency_random launch failed");
    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "dram_latency_random execution failed");

    register_latency<<<BLOCKS, THREAD_PER_BLOCK>>>(dummy, reg_delay);
    CHECK_CUDA_ERROR(cudaGetLastError(), "register_latency launch failed");
    CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "register_latency execution failed");

    std::ofstream csvFile("delay.csv");
    if (csvFile.is_open()) {
        for (int32_t i = 0; i < TOTAL_THREADS; i++) {
            csvFile << dram_delay_random[i] << "," << dram_delay_broadcast[i] << "," << reg_delay[i] 
                << "," << const_delay[i] << "\n";
        }
        csvFile << "dram_delay_random,dram_delay_broadcast,reg_delay,const_delay\n";
        csvFile.close();
    } else {
        std::cerr << "CSV file could not be opened.." << std::endl;
    }

    CHECK_CUDA_ERROR(cudaFree(data), "Failed to free data");
    CHECK_CUDA_ERROR(cudaFree(dummy), "Failed to free dummy");

    CHECK_CUDA_ERROR(cudaFree(dram_delay_random), "Failed to free dram_delay_random");
    CHECK_CUDA_ERROR(cudaFree(dram_delay_broadcast), "Failed to free dram_delay_broadcast");
    CHECK_CUDA_ERROR(cudaFree(reg_delay), "Failed to free reg_delay");
	CHECK_CUDA_ERROR(cudaFree(const_delay), "Failed to free const_delay");

    CHECK_CUDA_ERROR(cudaDeviceReset(), "Failed to reset device");

    return 0;
}