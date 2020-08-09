#include "Data.h"

using namespace cv;

void Data::load_images(const std::string& input)
{
	// loads in images and creates a standard vector of Mat objects 

	String path(input);
	std::vector<String> file_name; 
	std::vector<Mat> image_data_vector; 
	glob(path, file_name, true);

	for (size_t i = 0; i < file_name.size(); i++)
	{
		Mat im = imread(file_name[i], IMREAD_GRAYSCALE);				// read images in as grayscale 
		if (im.empty()) continue; 
		// include preprocessing here if necessary later 
		image_data_vector.push_back(im);

	} 
	Data::image_data = image_data_vector; 
	// std::cout << Data::image_data.size() << std::endl; 

	// cv::imshow("TEST", Data::image_data[1]);
	// cv::waitKey();
	
}

void Data::convert_images()
{

	cv::Mat image_1 = image_data[0]; 
	Eigen::MatrixXd X = Eigen::MatrixXd(image_1.rows, image_1.cols); 

	image_matrices.resize(image_data.size()); 

	for (int i = 0; i < image_data.size(); i++)
	{
		Eigen::MatrixXd temp_matrix = Eigen::MatrixXd(image_data[i].rows, image_data[i].cols); 
		image_matrices[i] = temp_matrix; 
	}

	std::cout << image_matrices[0].rows() << ", " << image_matrices[0].cols() << std::endl; 
}

void Data::set_label()
{
	// creates a standard vector of enumerated labels that are consistent with the file (training "cellphone-YES") 
	std::vector<int> label_data_vector; 
	for (int j = 0; j < image_data.size(); j++)
	{
		label_data_vector.push_back(j); 
	}

	image_labels = label_data_vector; 
}

