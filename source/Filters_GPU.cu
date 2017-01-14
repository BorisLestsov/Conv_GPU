/*
 * Filters_GPU.cu
 *
 *  Created on: Jan 8, 2017
 *      Author: boris
 */

#include "Filters.h"

__global__ void compute(
		unsigned int res_rows,
		unsigned int res_cols,
		unsigned char* img,
		unsigned char* res,
		unsigned int hor_radius,
		unsigned int ver_radius,
		float* ker)
{
	__shared__ float prod_cache[17*17*3];
	__shared__ float cache2[17*3];

	unsigned int img_cols = res_cols+2*hor_radius;
	unsigned int hor_size = 2*hor_radius + 1;
	unsigned int i, j;
	float pix_res = 0.0;

	prod_cache[(threadIdx.x*hor_size + threadIdx.y)*3 + threadIdx.z] =
			img[((blockIdx.x + threadIdx.x)*img_cols + blockIdx.y + threadIdx.y)*3
			    + threadIdx.z] * ker[threadIdx.x*hor_size + threadIdx.y];

	__syncthreads();

	if (threadIdx.y == 0){
		cache2[threadIdx.x*3 + threadIdx.z] = 0.0;
		for (j = threadIdx.x*hor_size*3 + threadIdx.z; j < (threadIdx.x+1)*hor_size*3; j+=3)
			cache2[threadIdx.x*3 + threadIdx.z] += prod_cache[j];
	}

	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0)
		for (i = threadIdx.z; i < hor_size*3; i+=3)
			pix_res += cache2[i];

	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0){
		res[(blockIdx.x*res_cols + blockIdx.y)*3 + threadIdx.z] = (unsigned char) pix_res;
	}

	//res[(blockIdx.x*res_cols + blockIdx.y)*3 + threadIdx.z] =
	//	img[((blockIdx.x + threadIdx.x)*img_cols + blockIdx.y + threadIdx.y)*3 + threadIdx.z]
	//		*ker[threadIdx.x*hor_size + threadIdx.y];
}
