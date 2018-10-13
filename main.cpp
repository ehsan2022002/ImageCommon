#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include "Test.h"

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

int main()
{
	
	Test t;
	
	/////t.autocrop("c://app//z.jpg"); 
	////////////////////////////////
	//src = imread("c://app//744.bmp",1);
  
	Mat imgA = imread("c://app//doc3//993.bmp");
	Mat imgB = imread("c://app//doc3//1005.bmp");

	
	//
	//

	Mat sceneP = imread("c://app//doc3//72.bmp");	
	Mat outImg;

	Mat find = imread("c://app//nat_1.jpg");	
	Mat label = imread("c://app//doc3//71.bmp");
	Mat label2 = imread("c://app//doc3//1051.bmp");
	outImg = sceneP;
	
	
	if(find.empty() && sceneP.empty())
	{
		cout << "Could not load the image!" << endl;		
		exit(0);
	}
	else
	{
		cout << "Images loaded succesfully" << endl;
	}

	//t.findObject( label, 500, Scalar(255,0,0) );
	//t.findObject( label2, 500, Scalar(0,255,0) );
	t.findObject ( find, 1500, Scalar(255,0,0) );			
	t.findObject ( find, 1500, Scalar(0,255,0) );			
	t.findObject ( find, 1500, Scalar(0,0,255) );			


	namedWindow("Match");
	imshow("Match",outImg);


	waitKey(0);
	
}

