#ifndef __CONVLAYER_h
#define __CONVLAYER_h

#include <Eigen/core>
#include <iostream> 
#include <vector> 

class ConvLayer
{
private:
	// constructor definition 
	size_t input_dim_x;
	size_t input_dim_y;
	size_t input_dim_c;
	size_t filter_dim_x;
	size_t filter_dim_y;
	size_t stride_x;
	size_t stride_y;
	size_t num_filters;


public:
	// member variables 
	std::vector<Eigen::MatrixXd> delta_filters;
	std::vector<Eigen::MatrixXd> output;
	std::vector<double> accumulated_loss_values; 

	ConvLayer(size_t Input_dim_x,
		size_t Input_dim_y,
		size_t Input_dim_c,
		size_t Filter_dim_x,
		size_t Filter_dim_y,
		size_t Stride_x,
		size_t Stride_y,
		size_t Num_filters);
	~ConvLayer(); 
	int GetOutputDimX();
	int GetOutputDimY(); 

	void Forward(Eigen::MatrixXd& input, std::vector<Eigen::MatrixXd>& kernel);			// returns vector of conv matrices, size = num_filters
	void Backprop(std::vector<Eigen::MatrixXd>& delta_matrix, Eigen::MatrixXd& input);

};

#endif
