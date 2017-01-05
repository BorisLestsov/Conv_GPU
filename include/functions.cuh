#ifndef _FUNCTIONS_CUH
#define _FUNCTIONS_CUH


#include <iostream>
#include <cuda_runtime.h>

#include "EasyBMP.h"

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)


#endif
