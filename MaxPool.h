#ifndef __MAXPOOL_H
#define __MAXPOOL_H 

#include <Eigen/Dense>
#include <vector>
#include <iostream> 

// i think this layer is fully good to implement as we don't really have to worry about channels i dont think. if we have to worry about channels here
// this will have to be revisited 

class MaxPool
{
private: 
	size_t input_dim_x;
	size_t input_dim_y;
	size_t input_dim_c;
	size_t filter_dim_x;
	size_t filter_dim_y;
	size_t stride_x;
	size_t stride_y;

public: 
	MaxPool(size_t Input_dim_x,
		size_t Input_dim_y,
		size_t Input_dim_c,
		size_t Filter_dim_x,
		size_t Filter_dim_y,
		size_t Stride_x,
		size_t Stride_y);

	~MaxPool(); 

	int GetOutputDimX();
	int GetOutputDimY();

	std::vector<Eigen::MatrixXd> output;					// final 
	std::vector<Eigen::MatrixXd> delta_matrices; 

	void Forward(std::vector<Eigen::MatrixXd>& input);		// final 
	void Backprop(std::vector<Eigen::MatrixXd>& delta_prev, std::vector<Eigen::MatrixXd>& prev_input); // to be revised... ugh 

};
#endif // !__MAXPOOL_H
