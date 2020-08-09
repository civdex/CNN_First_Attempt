#include "MaxPool.h"

MaxPool::MaxPool(size_t Input_dim_x,
	size_t Input_dim_y,
	size_t Input_dim_c,
	size_t Filter_dim_x,
	size_t Filter_dim_y,
	size_t Stride_x,
	size_t Stride_y)

{
	input_dim_x = Input_dim_x; 
	input_dim_y = Input_dim_y;
	input_dim_c = Input_dim_c; 
	filter_dim_x = Filter_dim_x; 
	filter_dim_y = Filter_dim_y;
	stride_x = Stride_x;
	stride_y = Stride_y;
}

MaxPool::~MaxPool()
{
	// destructor definition 
}

int MaxPool::GetOutputDimX()
{
	int output_dim_x_forward = (input_dim_x - filter_dim_x) / stride_x + 1;
	return output_dim_x_forward;
}

int MaxPool::GetOutputDimY()
{
	int output_dim_y_forward = (input_dim_y - filter_dim_y) / stride_y + 1;
	return output_dim_y_forward;
}

void MaxPool::Forward(std::vector<Eigen::MatrixXd>& input)
{
	// resize output matrices from conv operation to be consistent with the number of input matrices it receives 
	output.resize(input.size());

	// get size of output dimension
	int output_dim_x = (input_dim_x - filter_dim_x) / stride_x + 1;
	int output_dim_y = (input_dim_y - filter_dim_y) / stride_y + 1;

	// initialize output matrix and vectors
	Eigen::MatrixXd output_matrix;
	output_matrix.resize(output_dim_x, output_dim_y);

	for (size_t i = 0; i < input.size(); i++)				// iterate through the std::vector of Eigen Matrices 
	{
		int index_y = 0;									// y index for iterating through the input matrix
		int output_y = 0;									// y index for iterating through the ouput matrix
		while (index_y + filter_dim_y <= input_dim_y)
		{
			int index_x = 0;								// x index for iterating through the input matrix
			int output_x = 0;								// x index for iterating through the ouput matrix
			while (index_x + filter_dim_x <= input_dim_x)
			{
				output_matrix(output_x, output_y) = (input[i].block(index_x, index_y, filter_dim_x, filter_dim_y)).maxCoeff();
				index_x += stride_x;
				output_x += 1;
			}
			index_y += stride_y;
			output_y += 1;
		}
		output[i] = output_matrix;
	}

	std::cout << "\nMaxpooling: \n" << std::endl;

	/*
	for (int i = 0; i < output.size(); i++)
	{
		std::cout << output[i] << "\n\n" << std::endl;
	}
	*/ 

}

void MaxPool::Backprop(std::vector<Eigen::MatrixXd>& delta_prev, std::vector<Eigen::MatrixXd>& prev_input)
{

	// initialize int detla matrices (the matrices for just 1s and 0s to be multipled by the delta_prev
	std::vector <Eigen::MatrixXd> int_delta_matrices;
	int_delta_matrices.resize(prev_input.size());
	for (int i = 0; i < prev_input.size(); i++)
	{
		int_delta_matrices[i].resize(prev_input[i].rows(), prev_input[i].cols());
		for (int j = 0; j < int_delta_matrices[i].rows(); j++)
		{
			for (int k = 0; k < int_delta_matrices[i].cols(); k++)
			{
				int_delta_matrices[i](j, k) = 0;										// initialize as a matrix of 0s 
			}
		}
	}

	// NEW METHOD delete if necessary and it should go back to normal 
	for (int i = 0; i < prev_input.size(); i++)
	{
		int index_y = 0;
		int output_y = 0; 
		while (index_y + filter_dim_y <= prev_input[i].rows())
		{
			int index_x = 0;
			int output_x = 0; 
			while (index_x + filter_dim_x <= prev_input[i].cols())
			{
				Eigen::MatrixXd::Index maxRow, maxCol;
				double max_value = (prev_input[i].block(index_y, index_x, filter_dim_y, filter_dim_x)).maxCoeff(&maxRow, &maxCol);		// get indices
				
				int_delta_matrices[i](maxRow + index_y, maxCol + index_x) = 1 * delta_prev[i](output_y,output_x);		// input deltas to correct indices - delete the "delta_prev[i](output_y, outputx) to just get the 0s and 1s matrix		
				
				index_x += stride_x;
				output_x += 1;
			}
			index_y += stride_y;
			output_y += 1;
		}
	}

	delta_matrices = int_delta_matrices;

	/*
	std::cout << "The intermediate backprop maxpool is: \n" << std::endl;
	for (int i = 0; i < int_delta_matrices.size(); i++)
	{
		std::cout << "\n" << int_delta_matrices[i] <<"\n\n" << std::endl;
	}
	*/ 
}






