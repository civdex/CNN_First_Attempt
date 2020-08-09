#include "ConvLayer.h"

ConvLayer::ConvLayer(size_t Input_dim_x,
					size_t Input_dim_y,
					size_t Input_dim_c,
					size_t Filter_dim_x,
					size_t Filter_dim_y,
					size_t Stride_x,
					size_t Stride_y,
					size_t Num_filters)
{
	// constructor definitions 
	input_dim_x = Input_dim_x;
	input_dim_y = Input_dim_y;
	input_dim_c = Input_dim_c;
	filter_dim_x = Filter_dim_x;
	filter_dim_y = Filter_dim_y;
	stride_x = Stride_x;
	stride_y = Stride_y;
	num_filters = Num_filters;
}

ConvLayer::~ConvLayer()
{
	// deconstructor definition 
}

int ConvLayer::GetOutputDimX()
{
	int output_dim_x_forward = (input_dim_x - filter_dim_x) / stride_x + 1;
	return output_dim_x_forward;
}

int ConvLayer::GetOutputDimY()
{
	int output_dim_y_forward = (input_dim_y - filter_dim_y) / stride_y + 1;
	return output_dim_y_forward; 
}

void ConvLayer::Forward(Eigen::MatrixXd& input, std::vector<Eigen::MatrixXd>& kernel)
{
	// resize the ouput matrices from the conv operatoin to be consistent with the number of filters 
	output.resize(num_filters);

	// get size of output dimension
	int output_dim_x = (input_dim_x - filter_dim_x) / stride_x + 1;
	int output_dim_y = (input_dim_y - filter_dim_y) / stride_y + 1;

	// initialize output matrix and vectors
	Eigen::MatrixXd output_matrix;
	output_matrix.resize(output_dim_x, output_dim_y);	
	
																			
	// output_singleimage.resize(num_filters);	// output = vector of convoluted matrices, size = size of num filter

	for (int i = 0; i < num_filters; i++)
	{
		int index_y = 0;									// y index for iterating through the input matrix
		int output_y = 0;									// y index for iterating through the ouput matrix
		while (index_y + filter_dim_y <= input_dim_y)
		{
			int index_x = 0;								// x index for iterating through the input matrix
			int output_x = 0;								// x index for iterating through the ouput matrix
			while (index_x + filter_dim_x <= input_dim_x)
			{
				output_matrix(output_x, output_y) = (input.block(index_x, index_y, filter_dim_x, filter_dim_y).cwiseProduct(kernel[i])).sum();
				index_x += stride_x;
				output_x += 1;
			}
			index_y += stride_y;
			output_y += 1;
		}
		output[i] = (output_matrix);			// store the output  
	}

	std::cout << "\nConvolution: \n" << std::endl;
	/*
	for (int i = 0; i < output.size(); i++)
	{
		std::cout << output[i] << "\n\n" << std::endl;
	}
	*/ 

}

void ConvLayer::Backprop(std::vector<Eigen::MatrixXd>& delta_matrix, Eigen::MatrixXd& input)
{
	// get size of output dimension
	int output_dim_x = filter_dim_x;
	int output_dim_y = filter_dim_y;

	// size of matrix: 
	int delta_dim_x = delta_matrix[0].cols();
	int delta_dim_y = delta_matrix[0].rows(); 

	// initialize output matrix and vectors
	Eigen::MatrixXd kernel_matrix;
	kernel_matrix.resize(output_dim_x, output_dim_y);
	delta_filters.resize(num_filters);

	// input.resize(num_filters + 1);
	// delta_matrix.resize(num_filters + 1);  -- NOT SURE IF THIS SHOULD HAVE BEEN COMMENTED OUT DONE!!!!! 

	for (int i = 0; i < num_filters; i++)
	{
		int index_y = 0;									// y index for iterating through the input matrix
		int output_y = 0;									// y index for iterating through the ouput matrix
		while (index_y + delta_dim_y <= input_dim_y)
		{
			int index_x = 0;								// x index for iterating through the input matrix
			int output_x = 0;								// x index for iterating through the ouput matrix
			while (index_x + delta_dim_x <= input_dim_x)
			{
				kernel_matrix(output_x, output_y) = (input.block(index_x, index_y, delta_dim_x, delta_dim_y).cwiseProduct(delta_matrix[i])).sum();
				index_x += stride_x;
				output_x += 1;
			}
			index_y += stride_y;
			output_y += 1;
		}
		delta_filters[i] = kernel_matrix;					// store the output 
	}

	/*
	std::cout << "The new filters are shown below: " << std::endl; 
	for (int i = 0; i < delta_filters.size(); i++)
	{
		std::cout << delta_filters[i] << "\n" <<  std::endl;
	}
	*/


}

