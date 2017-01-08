#include "Filters.h"
#include "functions.cuh"

#include <cuda_runtime.h>

UnnormalizedFilter::UnnormalizedFilter(
		const FilterKernel & kernel_p,
		bool isGPU_f,
		bool check_range):

        kernel (kernel_p),
        hor_radius ( kernel_p.n_cols / 2),
		vert_radius ( kernel_p.n_rows / 2),
		isGPU(isGPU_f),
        check(check_range)
{
	if (isGPU)
		_conv_function = &UnnormalizedFilter::_conv_GPU;
	else
		_conv_function = &UnnormalizedFilter::_conv_CPU;
}


Pixel UnnormalizedFilter::_conv_GPU(const Image &m) const {
	return make_tuple(0, 0, 0);
}

Pixel UnnormalizedFilter::_conv_CPU(const Image &m) const {
	double r, g, b, sum_r = 0, sum_g = 0, sum_b = 0;

	for (uint i = 0; i < kernel.n_rows; ++i) {
		for (uint j = 0; j < kernel.n_cols; ++j) {
			r = static_cast<double>(get<0>(m(i, j)));
			g = static_cast<double>(get<1>(m(i, j)));
			b = static_cast<double>(get<2>(m(i, j)));
			r = r * kernel(i, j);
			g = g * kernel(i, j);
			b = b * kernel(i, j);
			sum_r += r;
			sum_g += g;
			sum_b += b;
		}
	}
	if (check){
		if (sum_r > 255)
			sum_r = 255;
		else if (sum_r < 0)
			sum_r = 0;

		if (sum_g > 255)
			sum_g = 255;
		else if (sum_g < 0)
			sum_g = 0;

		if (sum_b > 255)
			sum_b = 255;
		else if (sum_b < 0)
			sum_b = 0;
	}
	return make_tuple(sum_r, sum_g, sum_b);
}

Pixel UnnormalizedFilter::operator () (const Image &m) const {
    return (this->*_conv_function)(m);
}


Image UnnormalizedFilter::convolve(const Image& img) const {
	unsigned char* d_img, *d_res;
	unsigned char* img_raw, *img_res;

	img_raw = (unsigned char*) malloc(img.n_rows*img.n_cols*3);
	img_res = (unsigned char*) malloc(img.n_rows*img.n_cols*3);

	CUDA_CHECK_RETURN( cudaMalloc((void**) &d_img, img.n_rows*img.n_cols*3) );
	CUDA_CHECK_RETURN( cudaMalloc((void**) &d_res, img.n_rows*img.n_cols*3) );

	for (uint i = 0; i < img.n_rows; ++i){
		for (uint j = 0; j < img.n_cols; ++j){
			img_raw[(i*img.n_cols + j)*3 + 0] = get<0>(img(i, j));
			img_raw[(i*img.n_cols + j)*3 + 1] = get<1>(img(i, j));
			img_raw[(i*img.n_cols + j)*3 + 2] = get<2>(img(i, j));
		}
	}

	CUDA_CHECK_RETURN( cudaMemcpy(d_img, img_raw,
			img.n_rows*img.n_cols*3,
			cudaMemcpyHostToDevice)
			);


	//dim3 block_grid(n_rows, n_cols, 3);
	//dim2 thread_grid(n_rows, n_cols);
	//kernel<<<grid, thread_grid>>>(img.n_cols, d_img, d_res);
	dim3 block_grid(img.n_rows, img.n_cols);
	compute<<<block_grid, 3>>>(img.n_cols, d_img, d_res);


	CUDA_CHECK_RETURN( cudaMemcpy(img_res, d_res,
				img.n_rows*img.n_cols*3,
				cudaMemcpyDeviceToHost)
				);

	Image res(img.n_rows,  img.n_cols);

	for (uint i = 0; i < img.n_rows; ++i){
		for (uint j = 0; j < img.n_cols; ++j){
			get<0>(res(i, j)) = img_res[(i*img.n_cols + j)*3 + 0];
			get<1>(res(i, j)) = img_res[(i*img.n_cols + j)*3 + 1];
			get<2>(res(i, j)) = img_res[(i*img.n_cols + j)*3 + 2];
		}
	}

	CUDA_CHECK_RETURN( cudaFree(d_img) );
	CUDA_CHECK_RETURN( cudaFree(d_res) );

	free((void*) img_raw);
	free((void*) img_res);

	return res;
}



FloatFilter::FloatFilter(const FilterKernel & kernel_p):
    kernel (kernel_p),
    hor_radius (kernel_p.n_cols / 2),
    vert_radius (kernel_p.n_rows / 2)
{}

double FloatFilter::operator () (const Image &m) const
{
    uint hor_size = 2 * hor_radius + 1;
    uint vert_size = 2 * vert_radius + 1;
    double r, sum_r = 0;

    for (uint i = 0; i < vert_size; ++i) {
        for (uint j = 0; j < hor_size; ++j) {
            r = static_cast<double>(get<0>(m(i, j)));
            r = r * kernel(i, j);
            sum_r += r;
        }
    }
    return sum_r;
}


LBPFilter::LBPFilter():
        hor_radius (1),
        vert_radius (1)
{}

unsigned char LBPFilter::operator () (const Image& m) const
{
    char num = 0;
    uint cen_pix = get<0>(m(1,1));

    if (get<0>(m(1, 2)) >= cen_pix)
        num = num | R_MASK;
    if (get<0>(m(0, 2)) >= cen_pix)
        num = num | RU_MASK;
    if (get<0>(m(0, 1)) >= cen_pix)
        num = num | U_MASK;
    if (get<0>(m(0, 0)) >= cen_pix)
        num = num | LU_MASK;
    if (get<0>(m(1, 0)) >= cen_pix)
        num = num | L_MASK;
    if (get<0>(m(2, 0)) >= cen_pix)
        num = num | LD_MASK;
    if (get<0>(m(2, 1)) >= cen_pix)
        num = num | D_MASK;
    if (get<0>(m(2, 2)) >= cen_pix)
        num = num | RD_MASK;

    return num;
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
