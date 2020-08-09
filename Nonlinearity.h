#ifndef __NONLINEARITY_H
#define __NONLINEARITY_H

#include <vector>
#include <Eigen/Dense>
#include <math.h>
#include <iostream> 

class Nonlinearity
{
private:

	// some private members
	// flattened output 
	std::vector<Eigen::VectorXd> flattened_matrices;
	void Flatten(std::vector<Eigen::MatrixXd>& input);
	double sigmoid_i(double input); 

public: 
	// members - vector of standard vectors 
	std::vector<Eigen::VectorXd> sigmoid_output; 
	std::vector<Eigen::VectorXd> ReLU_output;

	std::vector<Eigen::MatrixXd> delta_sigmas;
	std::vector<Eigen::MatrixXd> delta_relus; 

	// member functions 

	void ReLU(std::vector<Eigen::MatrixXd>& input);
	void sigmoid(std::vector<Eigen::MatrixXd>& input);	// this should be good for now ffs 
	void Backprop_sigmoid(std::vector<Eigen::VectorXd>& delta_prev, std::vector<Eigen::MatrixXd>& prev_input);
	void Backprop_relu(std::vector<Eigen::VectorXd>& delta_prev, std::vector<Eigen::MatrixXd>& prev_input); 


};

#endif
