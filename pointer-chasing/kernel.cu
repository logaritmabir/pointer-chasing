#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <ctime>
#include <cstdint>
#include <fstream>
#include <stdio.h>
#include <curand_kernel.h>
#include <iomanip>

#define THREAD_PER_BLOCK 64
#define BLOCKS 16
#define TOTAL_THREADS (THREAD_PER_BLOCK * BLOCKS)
#define L1_CACHE_SIZE_IN_BYTES 64 * 1024
#define L2_CACHE_SIZE_IN_BYTES 1536 * 1024

#define CONSTANT_MEMORY_SIZE_IN_BYTES 32 * 1024 // Defination is 32 KB, but actual size is 64 KB
#define MAX_CONSTANT_MEMORY_ELEMENTS CONSTANT_MEMORY_SIZE_IN_BYTES / sizeof(int32_t)

#define ITERATIONS 1024

__device__ __constant__ int32_t d_const_data[MAX_CONSTANT_MEMORY_ELEMENTS];

__global__ void const_latency_contention(int32_t* dummy, float* d_const_delay_contention) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t result = 0;
	clock_t start = clock();
	for (int i = 0; i < ITERATIONS; i++) {
		result += d_const_data[0];
		result += d_const_data[1];
		result += d_const_data[2];
		result += d_const_data[3];
		result += d_const_data[4];
		result += d_const_data[5];
		result += d_const_data[6];
		result += d_const_data[7];
	}
	clock_t end = clock();
	dummy[idx] = result;
	d_const_delay_contention[idx] = static_cast<float>(end - start) / static_cast<float>(ITERATIONS);
}

__global__ void dram_latency_contention(int32_t* d_data, int32_t* dummy, float* d_dram_delay_contention) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t result = 0;
	clock_t start = clock();
	for (int i = 0; i < ITERATIONS; i++) {
		result += d_data[0];
		result += d_data[1];
		result += d_data[2];
		result += d_data[3];
		result += d_data[4];
		result += d_data[5];
		result += d_data[6];
		result += d_data[7];
	}
	clock_t end = clock();
	dummy[idx] = result;
	d_dram_delay_contention[idx] = static_cast<float>(end - start) / static_cast<float>(ITERATIONS);
}

__global__ void const_latency_random(int32_t* dummy, unsigned int seed, int32_t data_length, float* d_const_delay_random) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	curandState state;
	curand_init(seed + idx, 0, 0, &state);
	int32_t start_idx = curand(&state) % data_length;

	int32_t result = 0;

	clock_t start = clock();
	for (int i = 0; i < ITERATIONS; i++) {
		start_idx = d_const_data[start_idx];
		result += start_idx;
	}
	clock_t end = clock();
	dummy[idx] = result;
	d_const_delay_random[idx] = static_cast<float>(end - start) / static_cast<float>(ITERATIONS);
}

__global__ void const_latency_broadcast(int32_t* dummy, float* d_const_delay) {
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
	d_const_delay[idx] = static_cast<float>(end - start) / static_cast<float>(ITERATIONS);
}

__global__ void dram_latency_random(int32_t* data, int32_t* dummy, unsigned int seed, int32_t data_length, float* d_dram_delay_random) {
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
	d_dram_delay_random[idx] = static_cast<float>(end - start) / static_cast<float>(ITERATIONS);
}

__global__ void dram_latency_broadcast(int32_t* data, int32_t* dummy, float* d_dram_delay_broadcast) {
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
	d_dram_delay_broadcast[idx] = static_cast<float>(end - start) / static_cast<float>(ITERATIONS);
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
	//printDeviceProperties();
	const int32_t data_length = 256 * 1024 * 1024; // 1GB (uint32_t)

	int32_t* h_data, * h_const_data, * h_dummy;
	int32_t* d_data, * d_dummy;

	float* h_dram_delay_random, * h_dram_delay_broadcast, * h_reg_delay, * h_const_delay_broadcast, * h_const_delay_random,
		* h_dram_delay_contention, * h_const_delay_contention;
	float* d_dram_delay_random, * d_dram_delay_broadcast, * d_reg_delay, * d_const_delay_broadcast, * d_const_delay_random
		, * d_dram_delay_contention, * d_const_delay_contention;

	CHECK_CUDA_ERROR(cudaMalloc(&d_dummy, TOTAL_THREADS * sizeof(int32_t)), "Failed to allocate d_dummy");
	CHECK_CUDA_ERROR(cudaMalloc(&d_dram_delay_random, TOTAL_THREADS * sizeof(float)), "Failed to allocate d_dram_delay_random");
	CHECK_CUDA_ERROR(cudaMalloc(&d_dram_delay_broadcast, TOTAL_THREADS * sizeof(float)), "Failed to allocate d_dram_delay_broadcast");
	CHECK_CUDA_ERROR(cudaMalloc(&d_reg_delay, TOTAL_THREADS * sizeof(float)), "Failed to allocate d_reg_delay");
	CHECK_CUDA_ERROR(cudaMalloc(&d_const_delay_broadcast, TOTAL_THREADS * sizeof(float)), "Failed to allocate d_const_delay_broadcast");
	CHECK_CUDA_ERROR(cudaMalloc(&d_const_delay_random, TOTAL_THREADS * sizeof(float)), "Failed to allocate d_const_delay_random");

	h_data = new int32_t[data_length];
	for (int i = 0; i < data_length; i++) {
		h_data[i] = rand() % data_length;
	}

	h_const_data = new int32_t[MAX_CONSTANT_MEMORY_ELEMENTS];
	for (int i = 0; i < MAX_CONSTANT_MEMORY_ELEMENTS; i++) {
		h_const_data[i] = rand() % MAX_CONSTANT_MEMORY_ELEMENTS;
	}

	h_dummy = new int32_t[TOTAL_THREADS];
	h_dram_delay_random = new float[TOTAL_THREADS];
	h_dram_delay_broadcast = new float[TOTAL_THREADS];
	h_reg_delay = new float[TOTAL_THREADS];
	h_const_delay_broadcast = new float[TOTAL_THREADS];
	h_const_delay_random = new float[TOTAL_THREADS];
	h_const_delay_contention = new float[TOTAL_THREADS];
	h_dram_delay_contention = new float[TOTAL_THREADS];

	CHECK_CUDA_ERROR(cudaMemcpy(d_dummy, h_dummy, TOTAL_THREADS * sizeof(int32_t), cudaMemcpyHostToDevice), "Failed to copy h_dummy HtoD");
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_const_data, h_const_data, MAX_CONSTANT_MEMORY_ELEMENTS * sizeof(int32_t)), "Failed to copy const_data HtoD");
	const_latency_broadcast << <BLOCKS, THREAD_PER_BLOCK >> > (d_dummy, d_const_delay_broadcast);
	CHECK_CUDA_ERROR(cudaGetLastError(), "constant_memory_latency launch failed");
	CHECK_CUDA_ERROR(cudaMemcpy(h_const_delay_broadcast, d_const_delay_broadcast, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy const_delay DtoH");
	CHECK_CUDA_ERROR(cudaDeviceReset(), "Failed to reset device");

	CHECK_CUDA_ERROR(cudaMalloc(&d_data, data_length * sizeof(int32_t)), "Failed to allocate data");
	CHECK_CUDA_ERROR(cudaMalloc(&d_dummy, TOTAL_THREADS * sizeof(int32_t)), "Failed to allocate dummy");
	CHECK_CUDA_ERROR(cudaMalloc(&d_dram_delay_random, TOTAL_THREADS * sizeof(float)), "Failed to allocate dram_delay_random");
	CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_length * sizeof(int32_t), cudaMemcpyHostToDevice), "Failed to copy h_data HtoD");
	CHECK_CUDA_ERROR(cudaMemcpy(d_dummy, h_dummy, TOTAL_THREADS * sizeof(int32_t), cudaMemcpyHostToDevice), "Failed to copy h_dummy HtoD");
	CHECK_CUDA_ERROR(cudaMemcpy(d_dram_delay_random, h_dram_delay_random, TOTAL_THREADS * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy h_dram_delay_random HtoD");
	unsigned int seed = static_cast<unsigned int>(time(NULL));
	dram_latency_random << <BLOCKS, THREAD_PER_BLOCK >> > (d_data, d_dummy, seed, data_length, d_dram_delay_random);
	CHECK_CUDA_ERROR(cudaGetLastError(), "dram_latency_random launch failed");
	CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "dram_latency_random execution failed");
	CHECK_CUDA_ERROR(cudaMemcpy(h_dram_delay_random, d_dram_delay_random, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy dram_delay_random DtoH");
	CHECK_CUDA_ERROR(cudaDeviceReset(), "Failed to reset device");

	CHECK_CUDA_ERROR(cudaMalloc(&d_data, data_length * sizeof(int32_t)), "Failed to allocate d_data");
	CHECK_CUDA_ERROR(cudaMalloc(&d_dummy, TOTAL_THREADS * sizeof(int32_t)), "Failed to allocate d_dummy");
	CHECK_CUDA_ERROR(cudaMalloc(&d_dram_delay_broadcast, TOTAL_THREADS * sizeof(float)), "Failed to allocate d_dram_delay_broadcast");
	CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_length * sizeof(int32_t), cudaMemcpyHostToDevice), "Failed to allocate d_data");
	CHECK_CUDA_ERROR(cudaMemcpy(d_dummy, h_dummy, TOTAL_THREADS * sizeof(int32_t), cudaMemcpyHostToDevice), "Failed to allocate d_dummy");
	CHECK_CUDA_ERROR(cudaMemcpy(d_dram_delay_broadcast, h_dram_delay_broadcast, TOTAL_THREADS * sizeof(float), cudaMemcpyHostToDevice), "Failed to allocate d_dram_delay_broadcast");
	dram_latency_broadcast << <BLOCKS, THREAD_PER_BLOCK >> > (d_data, d_dummy, d_dram_delay_broadcast);
	CHECK_CUDA_ERROR(cudaGetLastError(), "dram_latency_broadcast launch failed");
	CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "dram_latency_broadcast execution failed");
	CHECK_CUDA_ERROR(cudaMemcpy(h_dram_delay_broadcast, d_dram_delay_broadcast, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy dram_delay_broadcast DtoH");
	CHECK_CUDA_ERROR(cudaDeviceReset(), "Failed to reset device");

	CHECK_CUDA_ERROR(cudaMalloc(&d_dummy, TOTAL_THREADS * sizeof(int32_t)), "Failed to allocate d_dummy");
	CHECK_CUDA_ERROR(cudaMalloc(&d_const_delay_random, TOTAL_THREADS * sizeof(float)), "Failed to allocate d_const_delay_random");
	CHECK_CUDA_ERROR(cudaMemcpy(d_dummy, h_dummy, TOTAL_THREADS * sizeof(int32_t), cudaMemcpyHostToDevice), "Failed to copy h_dummy HtoD");
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_const_data, h_const_data, MAX_CONSTANT_MEMORY_ELEMENTS * sizeof(int32_t)), "Failed to copy d_const_data HtoD");
	const_latency_random << <BLOCKS, THREAD_PER_BLOCK >> > (d_dummy, seed, MAX_CONSTANT_MEMORY_ELEMENTS, d_const_delay_random);
	CHECK_CUDA_ERROR(cudaGetLastError(), "constant_memory_latency launch failed");
	CHECK_CUDA_ERROR(cudaMemcpy(h_const_delay_random, d_const_delay_random, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy d_const_delay_random DtoH");
	CHECK_CUDA_ERROR(cudaDeviceReset(), "Failed to reset device");

	CHECK_CUDA_ERROR(cudaMalloc(&d_dummy, TOTAL_THREADS * sizeof(int32_t)), "Failed to allocate d_dummy");
	CHECK_CUDA_ERROR(cudaMalloc(&d_const_delay_contention, TOTAL_THREADS * sizeof(float)), "Failed to allocate d_const_delay_contention");
	CHECK_CUDA_ERROR(cudaMemcpy(d_dummy, h_dummy, TOTAL_THREADS * sizeof(int32_t), cudaMemcpyHostToDevice), "Failed to copy h_dummy HtoD");
	CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_const_data, h_const_data, MAX_CONSTANT_MEMORY_ELEMENTS * sizeof(int32_t)), "Failed to copy d_const_data HtoD");
	const_latency_contention << <BLOCKS, THREAD_PER_BLOCK >> > (d_dummy, d_const_delay_contention);
	CHECK_CUDA_ERROR(cudaGetLastError(), "constant_memory_latency launch failed");
	CHECK_CUDA_ERROR(cudaMemcpy(h_const_delay_contention, d_const_delay_contention, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy d_const_delay_contention DtoH");
	CHECK_CUDA_ERROR(cudaDeviceReset(), "Failed to reset device");

	CHECK_CUDA_ERROR(cudaMalloc(&d_data, data_length * sizeof(int32_t)), "Failed to allocate d_data");
	CHECK_CUDA_ERROR(cudaMalloc(&d_dummy, TOTAL_THREADS * sizeof(int32_t)), "Failed to allocate d_dummy");
	CHECK_CUDA_ERROR(cudaMalloc(&d_dram_delay_contention, TOTAL_THREADS * sizeof(float)), "Failed to allocate d_dram_delay_contention");
	CHECK_CUDA_ERROR(cudaMemcpy(d_dummy, h_dummy, TOTAL_THREADS * sizeof(int32_t), cudaMemcpyHostToDevice), "Failed to copy h_dummy HtoD");
	CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data, data_length * sizeof(int32_t), cudaMemcpyHostToDevice), "Failed to copy h_data HtoD");
	dram_latency_contention << <BLOCKS, THREAD_PER_BLOCK >> > (d_data, d_dummy, d_dram_delay_contention);
	CHECK_CUDA_ERROR(cudaGetLastError(), "dram_latency_contention launch failed");
	CHECK_CUDA_ERROR(cudaMemcpy(h_dram_delay_contention, d_dram_delay_contention, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy d_dram_delay_contention DtoH");
	CHECK_CUDA_ERROR(cudaDeviceReset(), "Failed to reset device");

	//register_latency<<<BLOCKS, THREAD_PER_BLOCK>>>(dummy);
	//CHECK_CUDA_ERROR(cudaGetLastError(), "register_latency launch failed");
	//CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "register_latency execution failed");
	//CHECK_CUDA_ERROR(cudaMemcpy(h_reg_delay, d_reg_delay, TOTAL_THREADS * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy reg_delay DtoH");

	std::ofstream csvFile("delay.csv");
	if (csvFile.is_open()) {
		csvFile << std::fixed << std::setprecision(8);
		for (int32_t i = 0; i < TOTAL_THREADS; i++) {
			csvFile << h_dram_delay_random[i] << "," << h_dram_delay_broadcast[i]
				<< "," << h_const_delay_random[i] << "," << h_const_delay_broadcast[i]
				<< "," << h_dram_delay_contention[i] << "," << h_const_delay_contention[i] << "\n";
		}
		csvFile << "dram_delay_random,dram_delay_broadcast,const_delay_random,const_delay_broadcast ,dram_delay_contention, const_delay_contention\n";
		csvFile.close();
	}
	else {
		std::cerr << "CSV file could not be opened.." << std::endl;
	}

	//CHECK_CUDA_ERROR(cudaFree(d_data), "Failed to free data");
	//CHECK_CUDA_ERROR(cudaFree(d_dummy), "Failed to free d_dummy");
	//CHECK_CUDA_ERROR(cudaFree(d_dram_delay_random), "Failed to free d_dram_delay_random");
	//CHECK_CUDA_ERROR(cudaFree(d_dram_delay_broadcast), "Failed to free d_dram_delay_broadcast");
	//CHECK_CUDA_ERROR(cudaFree(d_reg_delay), "Failed to free d_reg_delay");
	//CHECK_CUDA_ERROR(cudaFree(d_const_delay), "Failed to free d_const_delay");

	delete[] h_data;
	delete[] h_const_data;
	delete[] h_dummy;
	delete[] h_dram_delay_random;
	delete[] h_dram_delay_broadcast;
	delete[] h_reg_delay;
	delete[] h_const_delay_broadcast;

	CHECK_CUDA_ERROR(cudaDeviceReset(), "Failed to reset device");

	return 0;
}