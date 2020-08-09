#ifndef __DATA_H
#define __DATA_H 

#include <map>					// map class label to enumerated label
#include <unordered_set>		//keep track of the indexes as we split the data
#include <vector> 
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

// takes in the data and stores it in Mat objects 

class Data
{
private:  
	std::vector<int> image_labels;
	std::vector<cv::Mat> image_data;
public:
	void load_images(const std::string& input);		// function argument is file path within solution directory 
	void convert_images(); 
	void set_label(); 

	std::vector<Eigen::MatrixXd> image_matrices; 
};


#endif