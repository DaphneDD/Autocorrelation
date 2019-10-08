#!/bin/bash -e

filename="proj7b_results"

echo -e "CUDA\nmaxPerformance(megaMultSumsPerSecond)\n" > $filename

CUDA_PATH="/usr/local/apps/cuda/cuda-9.2"
CUDA_BIN_PATH="$CUDA_PATH/bin"
CUDA_NVCC="$CUDA_BIN_PATH/nvcc"

$CUDA_NVCC -o proj7b_cuda 7b_cuda.cu
./proj7b_cuda >>$filename


#openMP
echo -e "\n\nopenMP\nmaxPerformance(megaMultSumsPerSecond)\n" >> $filename
for t in 1 16
do
	g++ -DNUMT=$t 7b_omp.cpp -o proj7b_omp -fopenmp
	./proj7b_omp >> $filename
done

#SIMD
compiler=g++ 
$compiler -o simd.p4.o -c simd.p4.cpp
echo -e "\n\nSIMD\nmaxPerformance(megaMultSumsPerSecond)\n" >> $filename
$compiler -o proj7b_simd 7b_simd.cpp simd.p4.o -fopenmp
./proj7b_simd >> $filename
