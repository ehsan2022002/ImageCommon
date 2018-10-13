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
using namespace std;

#include "Test.h"

void Test::salt(cv::Mat &image, int n) 
{
	for (int k=0; k<n; k++) {
	// rand() is the MFC random number generator
	// try qrand() with Qt
	int i= rand()%image.cols;
	int j= rand()%image.rows;
	
	if (image.channels() == 1)
		{ // gray-level image
		image.at<uchar>(j,i)= 255;
		}
		else if (image.channels() == 3)
		{ // color image
		image.at<cv::Vec3b>(j,i)[0]= 255;
		image.at<cv::Vec3b>(j,i)[1]= 255;
		image.at<cv::Vec3b>(j,i)[2]= 255;
		}
	}
}


void Test::findObject( Mat objectP, int minHessian, Scalar color )
{

//read an image	
	Mat sceneP = imread("c://app//doc3//80.bmp");	
	Mat outImg;

	Mat find = imread("c://app//nat_1.jpg");	
	Mat label = imread("c://app//doc3//71.bmp");
	Mat label2 = imread("c://app//doc3//1051.bmp");
	outImg = sceneP;
	
	//vector of keypoints	
	vector<cv::KeyPoint> keypointsO;
	vector<cv::KeyPoint> keypointsS;
	cout << "Looking for object...\n";
	//Start the timer
	double duration;
	duration = static_cast<double>(cv::getTickCount());



	SurfFeatureDetector surf(minHessian);
	surf.detect(objectP,keypointsO);
	surf.detect(sceneP,keypointsS);


	//-- Step 2: Calculate descriptors (feature vectors)
  SurfDescriptorExtractor extractor;

  Mat descriptors_object, descriptors_scene;

  extractor.compute( objectP, keypointsO, descriptors_object );
  extractor.compute( sceneP, keypointsS, descriptors_scene );
  
 

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;  
  //BFMatcher matcher(NORM_L1);
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 150;
  double dist;

  //Quick calculation of min and max distances between keypoints
  for(int i=0; i<descriptors_object.rows; i++)
  {
	dist = matches[i].distance;
	if( dist < min_dist ) min_dist = dist;
	if( dist > max_dist ) max_dist = dist;
  }

 /* cout << "-- Max Dist: " << max_dist << endl;
  cout << "-- Min Dist: " << min_dist << endl;*/

  vector< DMatch > good_matches;

  for(int i = 0; i < descriptors_object.rows; i++)
  {
	  if( matches[i].distance < 3*min_dist) 
		  good_matches.push_back( matches[i] );
  }

	//drawMatches(objectP,keypointsO,sceneP,keypointsS,matches,outImg,Scalar::all(-1), Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	/*drawKeypoints(objectP,keypointsO,objectP,Scalar(0,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	namedWindow("SURF");
	imshow("SURF",objectP);*/

	//-- Localize the object

  if( good_matches.size() >=8 && good_matches.size() <= 30)
  {
	cout << "OBJECT FOUND!" << endl;
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for( unsigned int i = 0; i < good_matches.size(); i++ )
	{
		//-- Get the keypoints from the good matches
		obj.push_back( keypointsO[ good_matches[i].queryIdx ].pt );
	    scene.push_back( keypointsS[ good_matches[i].trainIdx ].pt );
	}

	Mat H = findHomography( obj, scene, CV_RANSAC );

  

	//-- Get the corners from the image_1 ( the object to be "detected" )
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( objectP.cols, 0 );
	obj_corners[2] = cvPoint( objectP.cols, objectP.rows ); obj_corners[3] = cvPoint( 0, objectP.rows );
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform( obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
  
	line( outImg, scene_corners[0] , scene_corners[1], Scalar(255,255,0), 2 ); //TOP line
	line( outImg, scene_corners[1] , scene_corners[2], color, 2 );
	line( outImg, scene_corners[2] , scene_corners[3], color, 2 );
	line( outImg, scene_corners[3] , scene_corners[0] , color, 2 );
  }
  else cout << "OBJECT NOT FOUND!" << endl;
	
	duration = static_cast<double>(cv::getTickCount())-duration;
	duration = (duration/cv::getTickFrequency()) * 1000;
	
	
	 
	//cout <<  << endl;
	cout << "Good matches found: " << good_matches.size() << endl;
	cout << "Algorithm duration: " << duration << endl << "--------------------------------------" << endl;


	// drawing the results
	namedWindow("matches");
	Mat img_matches;
	drawMatches(objectP, keypointsO, sceneP, keypointsS, good_matches, img_matches);
	imshow("matches", img_matches);
	waitKey(100);

}


void Test::colorReduce(cv::Mat &image, int div=64)
{
	int nl= image.rows; // number of lines
	// total number of elements per line
	int nc= image.cols * image.channels();
	for (int j=0; j<nl; j++) 
	{
		// get the address of row j
		uchar* data= image.ptr<uchar>(j);
		for (int i=0; i<nc; i++) 
		{
		// process each pixel ---------------------
		data[i]= data[i]/div*div + div/2;
		// end of pixel processing -----------
		} // end of line
	}

}


void Test::colorReduce2(cv::Mat &image, int div=64)
{
	// obtain iterator at initial position
	cv::Mat_<cv::Vec3b>::iterator it=image.begin<cv::Vec3b>();
	// obtain end position
	cv::Mat_<cv::Vec3b>::iterator itend=image.end<cv::Vec3b>();
	// loop over all pixels
	for ( ; it!= itend; ++it) 
	{
		// process each pixel ---------------------
		(*it)[0]= (*it)[0]/div*div + div/2;
		(*it)[1]= (*it)[1]/div*div + div/2;
		(*it)[2]= (*it)[2]/div*div + div/2;
		// end of pixel processing ----------------
	}
}

void Test::sharpen(const cv::Mat &image, cv::Mat &result)
{
	// allocate if necessary
	result.create(image.size(), image.type());
	for (int j= 1; j<image.rows-1; j++)
	{ // for all rows
		// (except first and last)
		const uchar* previous=image.ptr<const uchar>(j-1); // previous row
		const uchar* current=image.ptr<const uchar>(j); // current row
		const uchar* next=image.ptr<const uchar>(j+1); // next row
		uchar* output= result.ptr<uchar>(j); // output row
		for (int i=1; i<image.cols-1; i++)
		{
			*output++= cv::saturate_cast<uchar>(
			5*current[i]-current[i-1]
			-current[i+1]-previous[i]-next[i]);
		}
	}
		// Set the unprocess pixels to 0
		result.row(0).setTo(cv::Scalar(0));
		result.row(result.rows-1).setTo(cv::Scalar(0));
		result.col(0).setTo(cv::Scalar(0));
		result.col(result.cols-1).setTo(cv::Scalar(0));
}

void Test::sharpen2D(const cv::Mat &image, cv::Mat &result) 
{
	// Construct kernel (all entries initialized to 0)
	cv::Mat kernel(3,3,CV_32F,cv::Scalar(0));
	// assigns kernel values
	kernel.at<float>(1,1)= 5.0;
	kernel.at<float>(0,1)= -1.0;
	kernel.at<float>(2,1)= -1.0;
	kernel.at<float>(1,0)= -1.0;
	kernel.at<float>(1,2)= -1.0;
	//filter the image
	cv::filter2D(image,result,image.depth(),kernel);
}

cv::Mat Test::equalize(const cv::Mat &image)
{
	cv::Mat result;
	cv::equalizeHist(image,result);
	return result;
}

cv::Mat Test::getEdges(const cv::Mat &image,const int threshold) 
{
// Get the gradient image
	cv::Mat result;
	cv::morphologyEx(image,result,	cv::MORPH_GRADIENT,cv::Mat());
	// Apply threshold to obtain a binary image
	
	if (threshold>0)
		cv::threshold(result, result,threshold, 255, cv::THRESH_BINARY);

	return result;
}

void Test::faceDetect(const std::string ImageFileName)
{
	Mat image;
    image = imread(ImageFileName, CV_LOAD_IMAGE_COLOR); 
    namedWindow( "window1", 1 );  
	imshow( "window1", image );
 
    // Load Face cascade (.xml file)
	CascadeClassifier face_cascade("c:\\app\\haarcascade_frontalface_default.xml"); 
    //face_cascade.load ( "c:\\app\\haarcascade_frontalface_default.xml" );
 
    // Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
 
    // Draw circles on the detected faces
    for( int i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( image, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 255, 255 ), 4, 8, 0 );
    }
      
    imshow( "Detected Face", image );
     
    waitKey(0);                  
    
}
 
    
cv::Mat Test::rotate(Mat src, double angle)
{
    Mat dst;
    Point2f pt(src.cols/2., src.rows/2.);   
    Mat r = getRotationMatrix2D(pt, angle, 1.0);
    warpAffine(src, dst, r, Size(src.cols, src.rows));
    return dst;
}
	 
void Test::canny(string filename)
{
	Mat src1;
    src1 = imread(filename, CV_LOAD_IMAGE_COLOR);
    namedWindow( "Original image", CV_WINDOW_AUTOSIZE );
    imshow( "Original image", src1 );
 
    Mat gray, edge, draw;
    cvtColor(src1, gray, CV_BGR2GRAY);
 
    Canny( gray, edge, 50, 150, 3);
 
    edge.convertTo(draw, CV_8U);
    namedWindow("image", CV_WINDOW_AUTOSIZE);
    imshow("image", draw);
 
    waitKey(0);                                       
    
}


void Test::countContour()
{
	
	cv::Mat src = cv::imread("c://app//35_s.jpg");
	
	if (!src.data)
    return ;

	// Create binary image from source image
	cv::Mat bw;
	cv::cvtColor(src, bw, CV_BGR2GRAY);
	cv::threshold(bw, bw, 240, 255, CV_THRESH_BINARY);
	 

	cv::Mat dist;
	cv::distanceTransform(bw, dist, CV_DIST_L2, 3);
	imshow("A",dist);

	cv::normalize(dist, dist, 0, 1, cv::NORM_MINMAX);
	imshow("A2",dist);

	
	cv::threshold(dist, dist, .5, 1., CV_THRESH_BINARY);
	 
	// Create the CV_8U version of the distance image
	// It is needed for cv::findContours()
	cv::Mat dist_8u;
	dist.convertTo(dist_8u, CV_8U);

	// Find total markers
	std::vector<std::vector<cv::Point> > contours;
	cv::findContours(dist_8u, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	// Total objects
	int ncomp = contours.size();
	std::cout << ncomp <<std::endl;
	 
	int maxX = 0, minX = src.cols, maxY=0, minY = src.rows;

	for(int i=0; i<contours.size(); i++)
	{
    for(int j=0; j<contours[i].size(); j++)
    {
        Point p = contours[i][j];

        maxX = max(maxX, p.x);
        minX = min(minX, p.x);

        maxY = max(maxY, p.y);
        minY = min(minY, p.y);
    }

	rectangle( src, Point(minX,minY), Point(maxX, maxY), Scalar(0) );
	}

	imshow("trand4",src);
	imwrite("c://app//35_a.jpg",src);

	cvWaitKey(0);


}

void Test::autocrop(string filename)
{
    
	Mat src;
    src = imread(filename, CV_LOAD_IMAGE_COLOR);
    Mat gray;
    cvtColor(src, gray, CV_BGR2GRAY);
    threshold(gray, gray,200, 255,THRESH_BINARY_INV); //Threshold the gray
    //imshow("gray",gray);
	int largest_area=0;
    int largest_contour_index=0;
    Rect bounding_rect;
    vector<vector<Point>> contours; // Vector for storing contour
    vector<Vec4i> hierarchy;
    findContours( gray, contours, hierarchy,CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    // iterate through each contour.
    for( int i = 0; i< contours.size(); i++ )
    {
        //  Find the area of contour
        double a=contourArea( contours[i],false);
        if(a>largest_area){
            largest_area=a;
			//cout<<i<<" area  "<<a<<endl;
            // Store the index of largest contour
            largest_contour_index=i;              
            // Find the bounding rectangle for biggest contour
            bounding_rect=boundingRect(contours[i]);
        }
    }
    Scalar color( 255,255,255);  // color of the contour in the
    //Draw the contour and rectangle
    //drawContours( src, contours,largest_contour_index, color, CV_FILLED,8,hierarchy);
    //rectangle(src, bounding_rect,  Scalar(0,255,0),2, 8,0);
    namedWindow( "Display window", CV_WINDOW_AUTOSIZE );
    imshow( "Display window", src );    

	
	cv::Mat cropped = src(bounding_rect).clone();
	imshow("end",cropped);

}