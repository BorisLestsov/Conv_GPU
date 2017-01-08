#pragma once

#include "matrix.h"
#include "io.h"

using std::cout;
using std::endl;

using std::tuple;
using std::get;
using std::make_tuple;


typedef Matrix<double> FilterKernel;

class UnnormalizedFilter {
    FilterKernel kernel;

    const int isGPU;
    const int check;

    Pixel _conv_CPU (const Image &m) const;
    Pixel _conv_GPU (const Image &m) const;

    Pixel (UnnormalizedFilter::*_conv_function)(const Image &m) const;

public:
    const int hor_radius;
    const int vert_radius;

    UnnormalizedFilter(const FilterKernel & kernel_p, bool isGPU, bool check_range = true);

    Pixel operator () (const Image &m) const;

    Image convolve(const Image& img) const;
};

__global__ void compute(unsigned int cols, unsigned char* img, unsigned char* res);



class FloatFilter{
    FilterKernel kernel;

public:
    const int hor_radius;
    const int vert_radius;

    FloatFilter(const FilterKernel & kernel_p);
    double operator() (const Image&) const;
};


// Filter for local binary patterns detection
class LBPFilter{
    FilterKernel kernel;
    // counter clock-wise
    const unsigned char R_MASK = 1;
    const unsigned char RU_MASK = 2;
    const unsigned char U_MASK = 4;
    const unsigned char LU_MASK = 8;
    const unsigned char L_MASK = 16;
    const unsigned char LD_MASK = 32;
    const unsigned char D_MASK = 64;
    const unsigned char RD_MASK = 128;

public:
    const int hor_radius;
    const int vert_radius;

    LBPFilter();
    unsigned char operator() (const Image&) const;
};

FilterKernel make_gaussian_kernel(double sigma, int radius);
