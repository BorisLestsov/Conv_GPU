/*
 * matrix_GPU.cuh
 *
 *  Created on: Jan 8, 2017
 *      Author: boris
 */


template<typename ValueT>
template<typename UnaryMatrixOperator>
Matrix<typename std::result_of<UnaryMatrixOperator(Matrix<ValueT>)>::type>
	Matrix<ValueT>::unary_map_GPU(UnaryMatrixOperator &op) const
{
	typedef typename std::result_of<UnaryMatrixOperator(Matrix<ValueT>)>::type ReturnT;
	if (n_cols * n_rows == 0)
		return Matrix<ReturnT>(0, 0);

	const auto kernel_vert_radius = op.vert_radius;
	const auto kernel_hor_radius = op.hor_radius;


	Matrix<ValueT> extra_image = extra_borders(kernel_vert_radius, kernel_hor_radius);

	Matrix<ReturnT> tmp = op.convolve(extra_image);

	return tmp;
}


template<typename ValueT>
template<typename UnaryMatrixOperator>
Matrix<typename std::result_of<UnaryMatrixOperator(Matrix<ValueT>)>::type>
	Matrix<ValueT>::unary_map_GPU(const UnaryMatrixOperator &op) const
	{
		typedef typename std::result_of<UnaryMatrixOperator(Matrix<ValueT>)>::type ReturnT;
		if (n_cols * n_rows == 0)
			return Matrix<ReturnT>(0, 0);

		const auto kernel_vert_radius = op.vert_radius;
		const auto kernel_hor_radius = op.hor_radius;


		Matrix<ValueT> extra_image = extra_borders(kernel_vert_radius, kernel_hor_radius);

		Matrix<ReturnT> tmp = op.convolve(extra_image);

		return tmp;
	}
