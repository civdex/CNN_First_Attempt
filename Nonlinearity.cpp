#include "Nonlinearity.h"


/* make you only instantiate one class per sigmoid/ReLU !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void Nonlinearity::Flatten(std::vector<Eigen::MatrixXd>& input)
{
	std::vector<Eigen::MatrixXd> transposed_matrices;
	transposed_matrices.resize(input.size());
	flattened_matrices.resize(transposed_matrices.size()); 

	for (int i = 0; i < input.size(); i++)
	{
		transposed_matrices[i] = input[i].transpose(); 
		Eigen::VectorXd flat_input = Eigen::Map<const Eigen::VectorXd>(transposed_matrices[i].data(), transposed_matrices[i].size());	// flatten as eigen vect
		flattened_matrices[i] = flat_input;																							// std::vector of flattened inputs in VectorXd form 
	}
}

double Nonlinearity::sigmoid_i(double input)
{
	return (1 / (1 + exp(-(input))));
}

void Nonlinearity::Backprop_sigmoid(std::vector<Eigen::VectorXd>& delta_prev, std::vector<Eigen::MatrixXd>& prev_input)
{
	// delta_prev: the delta vector from previous layer FC layer, should be like a 9x1 vector or sometihng
	// prev_input, is the input matrix that went into this nonlinearity so it should be the output from the max pooling layer if that came before this,
	// need it to map the size 
	// do something 
	std::cout << " Backprop through non lin has begun: " << std::endl; 
	std::vector<Eigen::MatrixXd> deflattened_matrices;
	deflattened_matrices.resize(prev_input.size()); 

	std::cout << " we have sized deflattened matrices" << std::endl; 
	for (int i = 0; i < delta_prev.size(); i++)
	{
		Eigen::MatrixXd de_flattened_input = Eigen::Map<const Eigen::MatrixXd>(delta_prev[i].data(), prev_input[i].rows(), prev_input[i].cols());
		deflattened_matrices[i] = de_flattened_input.transpose(); 
	}
	
	std::cout << "These are the deflattened matrices" << std::endl; 
	/*
	for (int i = 0; i < deflattened_matrices.size(); i++)
	{
		std::cout << deflattened_matrices[i] << "\n" << std::endl; 
	}
	*/ 

	delta_sigmas.resize(deflattened_matrices.size()); 

	for (int i = 0; i < deflattened_matrices.size(); i++)				// itterating through ALL the flattened matrices
	{
		Eigen::MatrixXd temp_matrix; 
		for (int j = 0; j < deflattened_matrices[i].rows(); j++)		// iterating through all the elements of the individual matrix
		{
			for (int k = 0; k < deflattened_matrices[i].cols(); k++)
			{
				temp_matrix.resize(deflattened_matrices[i].rows(), deflattened_matrices[i].cols());
				temp_matrix(j, k) = deflattened_matrices[i](j, k)* sigmoid_i(prev_input[i](j,k)) * (1 - sigmoid_i(prev_input[i](j, k)));
			}
		}
		delta_sigmas[i] = temp_matrix;
	}

	/*
	// printing the values for testing 
	std::cout << "The output of the deflattened matrices (including derivative of the sigmoid function): \n" << std::endl;
	for (int i = 0; i < delta_sigmas.size(); i++)
	{
		std::cout << delta_sigmas[i] << "\n" << std::endl;
	}
	*/ 
}

void Nonlinearity::sigmoid(std::vector<Eigen::MatrixXd>& input)
{
	Flatten(input);

	for (int i = 0; i < flattened_matrices.size(); i++)				// itterating through ALL the flattened matrices
	{

		for (int j = 0; j < flattened_matrices[i].size(); j++)		// iterating through all the elements of the individual matrix
		{
			flattened_matrices[i][j] = (1 / (1 + exp(-(flattened_matrices[i])[j])));
		}
	}
	sigmoid_output = flattened_matrices; // consider deleting this and either making flattened_vector public or replacing "flattened vector" variable name with sigmoid_output 

	std::cout << "\Nonlinearity: " << std::endl;
	
	/*
	for (int i = 0; i < sigmoid_output.size(); i++)
	{
		std::cout << "\n" << std::endl;
		for (int j = 0; j < sigmoid_output[i].size(); j++)
		{
			std::cout << sigmoid_output[i][j] << std::endl;
		}
	}
	*/ 
}


void Nonlinearity::Backprop_relu(std::vector<Eigen::VectorXd>& delta_prev, std::vector<Eigen::MatrixXd>& prev_input)
{
	std::vector<Eigen::MatrixXd> deflattened_matrices;
	for (int i = 0; i < delta_prev.size(); i++)
	{
		Eigen::MatrixXd de_flattened_input = Eigen::Map<const Eigen::MatrixXd>(delta_prev[i].data(), prev_input[i].rows(), prev_input[i].cols());
		deflattened_matrices.push_back(de_flattened_input.transpose());
	}

	/*
	std::cout << "These are the deflattened matrices" << std::endl;
	for (int i = 0; i < deflattened_matrices.size(); i++)
	{
		std::cout << deflattened_matrices[i] << "\n" << std::endl;
	}
	*/ 

	delta_relus.resize(deflattened_matrices.size());
	for (int i = 0; i < deflattened_matrices.size(); i++)				// itterating through ALL the flattened matrices
	{
		Eigen::MatrixXd temp_matrix;
		for (int j = 0; j < deflattened_matrices[i].rows(); j++)		// iterating through all the elements of the individual matrix
		{
			for (int k = 0; k < deflattened_matrices[i].cols(); k++)
			{
				temp_matrix.resize(deflattened_matrices[i].rows(), deflattened_matrices[i].cols());
				if (prev_input[i](j,k) >= 0)
				{
					temp_matrix(j, k) = 1 * deflattened_matrices[i](j, k);
				}
				else
				{
					temp_matrix(j, k) = 0; 
				}
			}
		}
		delta_relus[i] = temp_matrix;
	}

	// printing the values for testing 
	std::cout << "The output of the deflattened matrices (including derivative of the sigmoid function): \n" << std::endl;
	for (int i = 0; i < delta_relus.size(); i++)
	{
		std::cout << delta_relus[i] << "\n" << std::endl;
	}
}

void Nonlinearity::ReLU(std::vector<Eigen::MatrixXd>& input)
{
	/// <summary>
	///  ReLU = max(0,x) 
	/// </summary>
	/// <param name="input"></param>

	Flatten(input);

	for (int i = 0; i < flattened_matrices.size(); i++)
	{
		for (int j = 0; j < flattened_matrices[i].size(); j++)
		{
			if ((flattened_matrices[i])[j] >= 0)
			{
				// do nothing, maintain the value 
			}
			else
			{
				flattened_matrices[i][j] = 0;
			}
		}
	}
	ReLU_output = flattened_matrices; // consider deleting this and either making flattened_vector public or replacing "flattened vector" variable name with ReLU_output 

	for (int i = 0; i < ReLU_output.size(); i++)
	{
		printf("\nThe %i matrix is:  \n", i + 1); 
		for (int j = 0; j < ReLU_output[i].size(); j++)
		{
			std::cout << ReLU_output[i][j] << std::endl; 
		}
		printf("\nThe number of entries in this matrix are: %i\n", ReLU_output[i].size()); 
	}

}

