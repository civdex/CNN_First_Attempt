#include "SoftMax.h"

void SoftMax::SoftMaxLoss(Eigen::Vector<double, 2>& input, int& actual_class)
{

	// soft max function 
	double s_z_1 = exp(input[0]) / (exp(input[0]) + exp(input[1]));
	double s_z_2 = exp(input[1]) / (exp(input[0]) + exp(input[1]));

	predicted_value_for_image[0] = s_z_1;
	predicted_value_for_image[1] = s_z_2;

	// soft maxs loss or binary cross entropy or something 
	double loss_val = -(actual_class * log10(s_z_1) + (1.0 - actual_class) * log10(s_z_2));
	
	accumulated_loss_values.push_back(loss_val);

	// std::cout << "\n The predicted values: " << predicted_value_for_image[0] << ", " << predicted_value_for_image[1] << std::endl; 

	// std::cout << "\nThe loss value for the current image is: " << loss_val << std::endl; 
}

void SoftMax::SoftMaxLossTotal()
{
	double sum_value = 0; 
	for (int i = 0; i < accumulated_loss_values.size(); i++)
	{
		sum_value += accumulated_loss_values[i]; 
	}
	loss_for_dataset = sum_value / accumulated_loss_values.size(); 
}

void SoftMax::Backprop(Eigen::Vector<double, 2>& input, int& actual_class)
{
	// delta_matrix: the deltas calculated from the derivative of the loss functions 

	// derivative of loss function: 
	
	delta_loss = (-1) * (actual_class / (predicted_value_for_image[0]) - (1.0 - actual_class) / (1 - predicted_value_for_image[0]));
	
	// std::cout << "The derivative of the loss is: " << delta_loss << std::endl; 
	

	// derivative of softmax function: 

	double ds1_x_1 = predicted_value_for_image[0] * (1 - predicted_value_for_image[0]);
	double ds1_x_2 = -predicted_value_for_image[1] * predicted_value_for_image[1];
	double ds2_x_1 = -predicted_value_for_image[0] * predicted_value_for_image[0];
	double ds2_x_2 = predicted_value_for_image[1] * (1 - predicted_value_for_image[1]);

	delta_softmax[0] = delta_loss * (ds1_x_1 + ds2_x_1);
	delta_softmax[1] = delta_loss * (ds1_x_2 + ds2_x_2);  // this is probably so wrong lmfao.. 
	
	// std::cout << delta_softmax << std::endl;
	
}
