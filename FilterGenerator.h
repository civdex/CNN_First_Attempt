#ifndef __FILTERGENERATOR_H
#define __FILTERGENERATOR_H

#include <stdlib.h> 
#include <Eigen/core>
#include <vector>

 
class FilterGenerator
{
private:
	size_t num_filters;
	size_t filter_dim_x;
	size_t filter_dim_y; 
public: 
	FilterGenerator(size_t Num_filters, size_t Filter_dim_x, size_t Filter_dim_y);
	~FilterGenerator();

	std::vector<Eigen::MatrixXd> output_filters;

	void create_filters();				// returns a set of filters - corresponding output: "std::vector<Eigen::MatrixXd> output_filters" 
};





#endif