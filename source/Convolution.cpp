/*
 ============================================================================
 Name        : Convolution.cu
 Author      : Me
 Version     :
 Copyright   : Nope
 Description : CUDA convolution
 ============================================================================
 */

#include <stdexcept>

#include "functions.cuh"
#include "matrix.h"
#include "io.h"
#include "Filters.h"

int main(int argc, char* argv[]) {

	try {
		if (argc != 3)
			throw std::invalid_argument("Wrong arguments");

		Image img = load_image(argv[1]);

		std::cout << "width:   " << img.n_rows << std::endl;
		std::cout << "height:  " << img.n_cols << std::endl;

		Image m = img.unary_map_GPU(UnnormalizedFilter(make_gaussian_kernel(14, 4), false));
		cout << "Finished" << endl;

		save_image(m, argv[2]);
	}
	catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return 1;
	}

	return 0;
}


