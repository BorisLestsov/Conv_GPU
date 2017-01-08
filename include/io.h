#pragma once

#include "matrix.h"
#include "EasyBMP.h"

#include <tuple>

typedef std::tuple<unsigned char, unsigned char, unsigned char> Pixel;
typedef Matrix<Pixel> Image;

Image load_image(const char*);
void save_image(const Image&, const char*);
