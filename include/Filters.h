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

public:
    const int hor_radius;
    const int vert_radius;
    const int check;

    UnnormalizedFilter(const FilterKernel & kernel_p, bool check_range = true);

    tuple<uint, uint, uint> operator () (const Image &m) const;
};

class FloatFilter{
    FilterKernel kernel;

public:
    const int hor_radius;
    const int vert_radius;

    FloatFilter(const FilterKernel & kernel_p);
    double operator() (const Image&) const;
};

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
