#ifndef __FULLYCONNECTED_H
#define __FULLYCONNECTED_H 
#include <assert.h> 

#include <vector>
#include <stdlib.h> 
#include <Eigen/Dense>
#include <numeric>			/*std::accumulate*/
#include <iostream> 

class FullyConnected
{
private:
	// some private members
public: 

	void Forward(std::vector<Eigen::VectorXd>& input, std::vector<Eigen::MatrixXd>& weight_matrices);		// FINAL 
	void Backprop(Eigen::Vector<double, 2>& delta_prev, std::vector<Eigen::MatrixXd>& weight_matrices, std::vector<Eigen::VectorXd>& same_input_as_feedforward);

	Eigen::Vector<double, 2> output;						// FINAL 

	std::vector<Eigen::MatrixXd> delta_weights;		// weight update values/deltas 
	std::vector<Eigen::VectorXd> delta_FC;			// output delta matrix for next layer  
};


#endif // !__FULLYCONNECTED_H
