#ifndef __SOFTMAX_H
#define __SOFTMAX_H

#include <vector>			/* std::vector */
#include <math.h>			/* exp, log */
#include <Eigen/Dense>		/* Eigen::MatrixXd */
#include <iostream>			/* std::cout */
#include <numeric>			/* std::accumulate. */

class SoftMax
{
private: 
	// some private members

public: 
	void SoftMaxLoss(Eigen::Vector<double, 2>& input, int& actual_class);
	Eigen::Vector<double, 2> predicted_value_for_image;
	std::vector<double> accumulated_loss_values;
	double loss_for_dataset;


	void SoftMaxLossTotal();

	void Backprop(Eigen::Vector<double, 2>& input, int& actual_class);
	double delta_loss;

	Eigen::Vector<double, 2> delta_softmax; 
};


#endif // !__SOFTMAX_H
