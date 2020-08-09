#include "FullyConnected.h"

void FullyConnected::Forward(std::vector<Eigen::VectorXd>& input, std::vector<Eigen::MatrixXd>& weight_matrices)
{
	for (size_t i = 0; i < input.size() - 1; i++)
	{
		assert(input[i].rows() == weight_matrices[i].cols());						// for matrix algebra - ensure that input rows = weight matrix cols
		assert(input[i].size() == input[i + 1].size());								// ensure that all delineated matrices in the input matrix have the same size
	} 

	output << 0, 0; 
	
	for (int i = 0; i < input.size(); i++)
	{
		output += weight_matrices[i] * input[i]; 
	}
	std::cout << "\n\nThe output for the fully connected layer is: \n " << std::endl;

}

void FullyConnected::Backprop(Eigen::Vector<double, 2>& delta_prev, std::vector<Eigen::MatrixXd>& weight_matrices, std::vector<Eigen::VectorXd>& same_input_as_feedforward)
{
	// delta matrix output 

	delta_FC.resize(weight_matrices.size()); 

	std::cout << "THE DELTA MATRIX TO PROPAGATE TO THE NEXT LAYER IS: \n" << std::endl;
	/*
	for (int i = 0; i < weight_matrices.size(); i++)
	{
		delta_FC[i] = weight_matrices[i].transpose() * delta_prev;
		std::cout << delta_FC[i] << "\n" << std::endl;				// prints delta_matrix_FC (delta output for this layer) 
	}
	*/ 

	// weight update:
	delta_weights.resize(same_input_as_feedforward.size());
	std::cout << "THE WEIGHT DELTA/UPDATE MATRIX IS: " << std::endl;


	for (int i = 0; i < same_input_as_feedforward.size(); i++)
	{
		delta_weights[i] = delta_prev * same_input_as_feedforward[i].transpose(); 	// dL/dw for FC weights 
		// std::cout << delta_weights[i] << "\n" << std::endl;
	}
}