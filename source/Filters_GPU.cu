/*
 * Filters_GPU.cu
 *
 *  Created on: Jan 8, 2017
 *      Author: boris
 */

#include "Filters.h"
#include <math.h>

__global__ void compute(unsigned int cols, unsigned char* img, unsigned char* res){
	int pix_ind = (blockIdx.x*cols + blockIdx.y)*3;
	int ch_ind = pix_ind + threadIdx.x;
	res[ch_ind] =  0.299*img[pix_ind] + 0.587*img[pix_ind+1] + 0.114*img[pix_ind+2];
}
