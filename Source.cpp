#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/src/Core/ArithmeticSequence.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <string.h>
#include <iostream> 

#include "Data.h"
#include "ConvLayer.h"
#include "FilterGenerator.h"
#include "Nonlinearity.h"
#include "MaxPool.h"
#include "FullyConnected.h"
#include "SoftMax.h"



void train(std::vector<Eigen::MatrixXd>& TRAINING_DATA, int learning_rate)
{
	// create filters for convolutional layer 
	int num_filters = 2;
	int filter_dim_x = 10; 
	int filter_dim_y = 10; 

	FilterGenerator filters(num_filters,filter_dim_x,filter_dim_y);
	filters.create_filters(); 
	std::vector<Eigen::MatrixXd> conv_filters; 
	conv_filters = filters.output_filters; 
	
	// get input image dimensions
	int input_dim_x = TRAINING_DATA[0].cols(); 
	int input_dim_y = TRAINING_DATA[0].rows();
	int input_dim_c = 1; 
	int stride_x = 1;
	int stride_y = 1; 


	ConvLayer convlayer_1(input_dim_x, input_dim_y, input_dim_c, filter_dim_x, filter_dim_y, stride_x, stride_y, num_filters);

	// max pool dimensions setup 
	int conv_output_dim_x = convlayer_1.GetOutputDimX();
	int conv_output_dim_y = convlayer_1.GetOutputDimY();
	int m_filter_dim_x = 2;
	int m_filter_dim_y = 2; 
	int m_stride_x = 1; 
	int m_stride_y = 1; 
	MaxPool maxpool_1(conv_output_dim_x, conv_output_dim_y, input_dim_c, m_filter_dim_x, m_filter_dim_y, m_stride_x, m_stride_y);

	// nonlinearity 
	Nonlinearity sigmoid_1;

	// fully connected layer 
	FullyConnected fullyconnected_1; 
	int num_weight_matrices = num_filters; 
	int fc_filter_dim_x = 2;
	int fc_filter_dim_y = maxpool_1.GetOutputDimX() * maxpool_1.GetOutputDimY(); 

	FilterGenerator fc_weights_init(num_filters, fc_filter_dim_x, fc_filter_dim_y);
	fc_weights_init.create_filters(); 
	std::vector<Eigen::MatrixXd> fc_weights; 
	fc_weights = fc_weights_init.output_filters; 
	
	// soft max layer 
	SoftMax softmax; 

	std::vector<int> actual_class;
	for (int i = 0; i < 72; i++)
	{
		actual_class.push_back(1); 
	}

	int epoch_total = 1;
	int SIZE_OF_DATASET = TRAINING_DATA.size();

	for (int epoch = 0; epoch < epoch_total; epoch++)
	{
		for (int i = 0; i < TRAINING_DATA.size(); i++)
		{
			printf("\nFOR IMAGE %i: \n", i + 1);

			// FORWARD PROP

			convlayer_1.Forward(TRAINING_DATA[i], conv_filters);
			maxpool_1.Forward(convlayer_1.output);
			sigmoid_1.sigmoid(maxpool_1.output);
			fullyconnected_1.Forward(sigmoid_1.sigmoid_output, fc_weights);
			softmax.SoftMaxLoss(fullyconnected_1.output, actual_class[i]);
			
			// BACKPROP
			softmax.Backprop(fullyconnected_1.output, actual_class[i]);
			fullyconnected_1.Backprop(softmax.delta_softmax, fc_weights, sigmoid_1.sigmoid_output);
			sigmoid_1.Backprop_sigmoid(fullyconnected_1.delta_FC, maxpool_1.output); 
			maxpool_1.Backprop(sigmoid_1.delta_sigmas, convlayer_1.output); 
			convlayer_1.Backprop(maxpool_1.delta_matrices, TRAINING_DATA[i]);

			for (int i = 0; i < num_filters; i++)
			{
				fc_weights[i] = fc_weights[i] - learning_rate * fullyconnected_1.delta_weights[i];
				conv_filters[i] = conv_filters[i] - learning_rate * convlayer_1.delta_filters[i];
			}
			
		}
		softmax.SoftMaxLossTotal();																	// just calculates the loss
		std::cout << "\nThe loss for the dataset is: " << softmax.loss_for_dataset << std::endl; 

		/*
		if (epoch_total % epoch == 10)
		{
			printf("EPOCH:\t %i \tLOSS: %i", epoch, softmax.loss_for_dataset);
		}
		*/
		softmax.accumulated_loss_values.clear();										// clear values for next epoch 
		softmax.SoftMaxLossTotal();
		softmax.accumulated_loss_values = { 1, 2, 3 }; 
		std::cout << "\nThe loss for the dataset is: " << softmax.loss_for_dataset << std::endl;

		
	}



}

int main(int argv, char** argc)
{
	// TEST MATRICES FOR FEEDFORWARD: 

	Eigen::MatrixXd matrix_1;
	matrix_1.resize(6, 6);
	matrix_1 << 0.1, 0.2, 0.9, 0.4, 0.3, 0.4,
				0.8, 0.1, 0.4, 0.9, 0.6, 0.7,
				0.1, 0.4, 0.2, 0.6, 0.7, 0.8,
				0.3, 0.2, 0.2, 0.1, 0.3, 0.5,
				0.7, 0.9, 0.2, 0.4, 0.7, 0.1,
				0.8, 0.6, 0.5, 0.2, 0.3, 0.1;
	Eigen::MatrixXd matrix_2;
	matrix_2.resize(6, 6);
	matrix_2 << 0.1, 0.7, 0.9, 0.7, 0.3, 0.1,
				0.7, 0.9, 0.2, 0.4, 0.7, 0.1,
				0.1, 0.4, 0.2, 0.3, 0.7, 0.1,
				0.1, 0.1, 0.2, 0.2, 0.7, 0.1,
				0.8, 0.2, 0.4, 0.1, 0.1, 0.1,
				0.1, 0.8, 0.2, 0.1, 0.1, 0.1;
	Eigen::MatrixXd matrix_3;
	matrix_3.resize(6, 6);
	matrix_3 << 0.1, 0.2, 0.9, 0.4, 0.3, 0.1,
				0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
				0.1, 0.4, 0.3, 0.6, 0.7, 0.8,
				0.1, 0.3, 0.2, 0.6, 0.7, 0.3,
				0.8, 0.2, 0.4, 0.7, 0.3, 0.6,
				0.5, 0.2, 0.9, 0.4, 0.3, 0.4;

	std::vector<Eigen::MatrixXd> input_matrices;
	input_matrices.push_back(matrix_1);
	input_matrices.push_back(matrix_2);
	input_matrices.push_back(matrix_3);
	
	std::cout << "THE RESULTS OF TRAIN: " << std::endl; 

	

	// CALLING DATA HANDLER: 
	Data train_images_yes;

	train_images_yes.load_images("Cellphones/training/training/test/cracked/");
	train_images_yes.set_label();
	train_images_yes.convert_images();


	Data train_images_no;
	train_images_no.load_images("Cellphones/training/training/test/intact");
	train_images_no.set_label();
	
	train(train_images_yes.image_matrices, 0.5);


	return 0; 
}