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
	__shared__ float prod_cache[1024*3];

	unsigned int img_cols = res_cols+2*hor_radius;
	unsigned int hor_size = 2*hor_radius + 1;
	unsigned int i;
	float pix_res = 0.0;

	prod_cache[(threadIdx.x*hor_size + threadIdx.y)*3 + threadIdx.z] =
			img[((blockIdx.x + threadIdx.x)*img_cols + blockIdx.y + threadIdx.y)*3
			    + threadIdx.z] * ker[threadIdx.x*hor_size + threadIdx.y];

	__syncthreads();

	// FIXME
	if (threadIdx.x == 0 && threadIdx.y == 0){
		for (i = threadIdx.z; i < hor_size*hor_size*3; i+=3)
			pix_res += prod_cache[i];
		res[(blockIdx.x*res_cols + blockIdx.y)*3 + threadIdx.z] = (unsigned char) pix_res;
	}

	//res[(blockIdx.x*res_cols + blockIdx.y)*3 + threadIdx.z] =
	//	img[((blockIdx.x + threadIdx.x)*img_cols + blockIdx.y + threadIdx.y)*3 + threadIdx.z]
	//		*ker[threadIdx.x*hor_size + threadIdx.y];
}
