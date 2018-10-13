#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2\nonfree\features2d.hpp"
#include <opencv2\calib3d\calib3d.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;

class Test
{

	cv::Mat ima;
private:	
	
public:
	void Test::salt(cv::Mat &image, int n) ;
	void Test::colorReduce(cv::Mat &image, int div);
	void Test::colorReduce2(cv::Mat &image, int div);

	void Test::sharpen(const cv::Mat &image, cv::Mat &result);
	void Test::unsharpMask(cv::Mat& im);	
	void Test::sharpen2D(const cv::Mat &image, cv::Mat &result);
	cv::Mat Test::equalize(const cv::Mat &image);
		
	cv::Mat Test::getEdges(const cv::Mat &image,const int threshold) ;
	void Test::faceDetect(const std::string ImageFileName);	
	cv::Mat Test::rotate(cv::Mat src, double angle);
	void Test::canny(std::string filename);
	
	void Test::autocrop(std::string filename);
	void Test::countContour();
	void findObject( Mat objectP, int minHessian, Scalar color );



};
