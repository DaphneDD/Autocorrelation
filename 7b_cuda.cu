//This program calculates the autocorrelation of a given signal stored in signal.txt with CUDA
//Programmer: Xiaoqiong Dong
//Date: June 8, 2019


#include <cmath>
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <assert.h>
#include <malloc.h>
#include <stdio.h>
#include <fstream>


//CUDA runtime
#include <cuda_runtime.h>

//Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE	64		//number of threads per block
#endif


//how many tries to get an average performance
#ifndef NUMTRIES
#define NUMTRIES 10
#endif

//Setting whether to write out autocorrelation result
#ifndef WRITEAUTOCORR
#define WRITEAUTOCORR 0
#endif

using std::cout;
using std::endl;
using std::cerr;
using std::ofstream;


//function prototypes
float Ranf(float, float);
int Ranf(int, int);
void TimeOfDaySeed();

//calculate autocorrelation (CUDA KERNEL) on the device
__global__ void autocorr(float *array, float *blockSums, int *shift)
{
	__shared__ int blockProducts[BLOCKSIZE];
	unsigned int numItems = blockDim.x;
	unsigned int tnum = threadIdx.x;
	unsigned int wgNum = blockIdx.x;
	unsigned int gid = numItems * wgNum + tnum;
	
	//calculate a single product
	blockProducts[tnum] = array[gid] * array[gid + shift[0]];
	
	//calculate the sum of products in this block
	__syncthreads();
	if (tnum == 0)
	{
		for (int i=1; i<BLOCKSIZE; i++)
			blockProducts[tnum] += blockProducts[i];
		blockSums[wgNum] = blockProducts[tnum];
	}
}

int main(int argc, char *argv[])
{

	int dev = findCudaDevice(argc, (const char**) argv);
	
	//read in data
	FILE *fp = fopen("signal.txt", "r");
	if (fp == NULL)
	{
		cerr << "Cannot open file 'signal.txt'" << endl;
		exit(1);
	}
	int size;
	fscanf(fp, "%d", &size);
	float *h_array = new float[2*size];
	float *h_sums = new float[1*size];
	for (int i=0; i<size; i++)
	{
		fscanf(fp, "%f", &h_array[i]);
		h_array[i + size] = h_array[i];
		h_sums[i] = 0;
	}
	fclose(fp);

	//allocate host memory for storing the sums of products
	float *h_blockSums = new float[size/BLOCKSIZE];  // sums of products returned by each block
	

	//allocate device memory
	float *d_array, *d_blockSums;
	int *d_shift;
	dim3 dims_d_array(size*2, 1, 1);
	dim3 dims_d_blockSums(size/BLOCKSIZE, 1, 1);
	dim3 dims_d_shift(1, 1, 1);
	
	cudaError_t status;
	status = cudaMalloc(reinterpret_cast<void **>(&d_array), size*2*sizeof(float));
	checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void **>(&d_blockSums), size/BLOCKSIZE*sizeof(float));
	checkCudaErrors(status);
	status = cudaMalloc(reinterpret_cast<void **>(&d_shift), sizeof(int));
	checkCudaErrors(status);
	
	//copy host memory to the device
	status = cudaMemcpy(d_array, h_array, size*2*sizeof(float), cudaMemcpyHostToDevice);
	checkCudaErrors(status);
	
	//setup the execution parameters
	dim3 threads(BLOCKSIZE, 1, 1);
	dim3 grid(size/threads.x, 1, 1);
	
	//create and start timer
	cudaDeviceSynchronize();
	
	//allocate CUDA events that we'll use for timing
	cudaEvent_t start, stop;
	status = cudaEventCreate(&start);
	checkCudaErrors(status);
	status = cudaEventCreate(&stop);
	checkCudaErrors(status);
	
	//record the start event
	status = cudaEventRecord(start, NULL);
	checkCudaErrors(status);
	
	//execute the kernel
	for (int t=0; t<NUMTRIES; t++)
	{
		for (int shift=0; shift<size; shift++)
		{
			status = cudaMemcpy(d_shift, &shift, sizeof(int), cudaMemcpyHostToDevice);
			checkCudaErrors(status);
			autocorr <<< grid, threads >>> (d_array, d_blockSums, d_shift);
			
			//copy result from the device to the host
			status = cudaMemcpy(h_blockSums, d_blockSums, size/BLOCKSIZE*sizeof(float), cudaMemcpyDeviceToHost);
			checkCudaErrors(status);
			
			//calculate the autocorrelation
			for (int j=0; j<size/BLOCKSIZE; j++)
				h_sums[shift] += h_blockSums[j];
		}
	}
	
	//record the stop event
	status = cudaEventRecord(stop, NULL);
	checkCudaErrors(status);
	
	//wait for the stop event to complete
	status = cudaEventSynchronize(stop);
	checkCudaErrors(status);
	
	float msecTotal = 0.0f;
	status = cudaEventElapsedTime(&msecTotal, start, stop);
	checkCudaErrors(status);
	
	//compute and print the performance
	double secondsTotal = 0.001 * static_cast<double>(msecTotal);
	double multsPerSecond = static_cast<double>(size) * static_cast<double>(size) * 2.0 * static_cast<double>(NUMTRIES)/ secondsTotal;
	double megaMultsPerSecond = multsPerSecond / 1000000.;
	cout << size << '\t' << BLOCKSIZE << '\t' << megaMultsPerSecond << '\t'; 
	
	//clean up memory;
	delete [] h_array;
	delete [] h_blockSums;
	delete [] h_sums;
	
	status = cudaFree(d_array);
	checkCudaErrors(status);
	status = cudaFree(d_blockSums);
	checkCudaErrors(status);
	status = cudaFree(d_shift);
	checkCudaErrors(status);

	return 0;	
}

