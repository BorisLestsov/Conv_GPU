/*
 * functions.cu
 *
 *  Created on: Jan 4, 2017
 *      Author: boris
 */

#include "functions.cuh"

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}


FilterKernel make_gaussian_kernel(double sigma, int radius){
	if (radius < 0)
		throw std::invalid_argument("Wrong filter radius");

	uint size = radius * 2 + 1;
	FilterKernel gauss_kernel(size, size);

	double sum = 0.0;
	for (uint i = 0; i < size; ++i) {
		for (uint j = 0; j < size; ++j) {
			gauss_kernel(i, j) = std::exp(-0.5 * (std::pow((i - radius) / sigma, 2.0) + std::pow((j - radius) / sigma, 2.0)))
						   / (2 * M_PI * std::pow(sigma, 2.0));
			sum += gauss_kernel(i, j);
		}
	}

	for (uint i = 0; i < size; ++i)
		for (uint j = 0; j < size; ++j)
			gauss_kernel(i, j) /= sum;

	return gauss_kernel;
}
