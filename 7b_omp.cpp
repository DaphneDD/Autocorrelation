//This program calculates the autocorrelation of a given signal stored in signal.txt
//Programmer: Xiaoqiong Dong
//Date: June 6, 2019

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

//Setting the number of threads
#ifndef NUMT
#define NUMT 1
#endif

//Setting the number of tries to discover the maximum performance
#ifndef NUMTRIES
#define NUMTRIES 10
#endif

//Setting whether to write out autocorrelation result
#ifndef WRITEAUTOCORR
#define WRITEAUTOCORR 0
#endif

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
	}
	fclose(fp);
	
#ifndef _OPENMP
	cerr << "No OpenMP support!" << endl;
	exit(1);
#endif
	
	omp_set_num_threads(NUMT); // set the number of threads to use in the for loop
	float timeSpent = 0;
	float maxPerformance = 0; //mega multsum per second
	for (int t=0; t<NUMTRIES; t++)
	{
		double time0 = omp_get_wtime();
		#pragma omp parallel for default(none) shared(array, sums, size)
		for (int shift = 0; shift < size; shift++)
		{
			float sum = 0;
			for (int i=0; i<size; i++)
				sum += array[i] * array[i + shift];
			sums[shift] = sum;
		}
		double time1 = omp_get_wtime();
		timeSpent = time1 - time0;
		double megaMultSumsPerSecond = 2.0 * (double)(size) *(double)(size)/ timeSpent / 1000000.0;
		if (megaMultSumsPerSecond > maxPerformance)
			maxPerformance = megaMultSumsPerSecond;
	}
	
	cout << NUMT << '\t' << maxPerformance << endl;
	
	//write out autocorrelation results
	if (WRITEAUTOCORR)
	{
		ofstream output;
		output.open("autocorr_omp");
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
