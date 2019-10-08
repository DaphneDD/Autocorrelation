//This program calculates the autocorrelation of a given signal stored in signal.txt with SIMD
//Programmer: Xiaoqiong Dong
//Date: June 7, 2019

#include "simd.p4.h"
#include <cmath>
#include <iostream>
#include <omp.h>
#include <ctime>
#include <cstdlib>
#include <stdio.h>
#include <fstream>

using std::cout;
using std::endl;
using std::cerr;
using std::ofstream;

//Setting the number of tries to discover the maximum performance
#ifndef NUMTRIES
#define NUMTRIES 10
#endif

//Setting whether to write out autocorrelation result
#ifndef WRITEAUTOCORR
#define WRITEAUTOCORR 0
#endif

//for prefetching
#define WILL_READ_ONLY		0
#define WILL_READ_AND_WRITE	1
#define LOCALITY_NONE		0
#define LOCALITY_LOW		1
#define LOCALITY_MED		2
#define LOCALITY_HIGH		3
#define PD					32 	//prefetch distance (fp words)
#define ONETIME				16

int main()
{

	FILE *fp = fopen("signal.txt", "r");
	if (fp == NULL)
	{
		cerr << "Cannot open file 'signal.txt'" << endl;
		exit(1);
	}
	int size;
	fscanf(fp, "%d", &size);
	float *array = new float[2*size];
	float *sums = new float[1*size];
	for (int i=0; i<size; i++)
	{
		fscanf(fp, "%f", &array[i]);
		array[i + size] = array[i];
		sums[i] = 0;
	}
	fclose(fp);
	
#ifndef _OPENMP
	cerr << "No OpenMP support!" << endl;
	exit(1);
#endif

	float timeSpent = 0;
	float maxPerformance = 0; //mega multsum per second
	for (int t=0; t<NUMTRIES; t++)
	{
		double time0 = omp_get_wtime();
		for (int shift = 0; shift < size; shift ++)
		{
			for (int i=0; i < size; i += ONETIME)
			{
				__builtin_prefetch(&array[i+PD], WILL_READ_ONLY, LOCALITY_LOW);
				__builtin_prefetch(&array[i + shift + PD], WILL_READ_ONLY, LOCALITY_LOW);
				sums[shift] += SimdMulSum(&array[i], &array[i + shift], ONETIME);
			}
		}
		double time1 = omp_get_wtime();
		timeSpent = time1 - time0;
		double megaMultSumsPerSecond = 2.0 * (double)(size) *(double)(size)/ timeSpent / 1000000.0;
		if (megaMultSumsPerSecond > maxPerformance)
			maxPerformance = megaMultSumsPerSecond;
	}
	
	cout << maxPerformance << endl;
	
	//write out autocorrelation results
	if (WRITEAUTOCORR)
	{
		ofstream output;
		output.open("autocorr_simd");
		if (output.is_open())
		{
			for (int i=0; i<size; i++)
				output << sums[i] <<endl;
		}
		output.close();
	}
	
	delete [] array;
	delete [] sums;
}
