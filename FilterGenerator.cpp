#include "FilterGenerator.h"

FilterGenerator::FilterGenerator(size_t Num_filters, size_t Filter_dim_x, size_t Filter_dim_y)
{
	// constructor definition 
	num_filters = Num_filters;
	filter_dim_x = Filter_dim_x;
	filter_dim_y = Filter_dim_y; 

}

FilterGenerator::~FilterGenerator()
{
	// deconstructor definition 
}

void FilterGenerator::create_filters()
{
	std::vector<Eigen::MatrixXd> filters;
	filters.resize(num_filters); 

	for (int i = 0; i < num_filters; i++)
	{
		Eigen::MatrixXd temp_matrix = Eigen::MatrixXd::Random(filter_dim_x, filter_dim_y); 
		filters[i] = temp_matrix;  
	}
	output_filters = filters; 
}

