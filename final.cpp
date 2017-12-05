/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - final.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include<stdlib.h>
#include<math.h>
#include <fstream>

#define MAGTHR 200
#define MIN_CENTER_DIST 7.1

using namespace std;
using namespace cv;

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;
vector<Vec3f> concentricCircles;

vector< vector<int> > imagedata;




/** Function Declaration */
vector<Rect> ViolaJones( Mat frame);
void Loaddata(string filename);
//GadientMag function series
Mat GradientMag(Mat src);
void conv(cv::Mat &input, cv::Mat &kernel, int size, cv::Mat &blurredOutput);
void magnitude(cv::Mat &x, cv::Mat &y, cv::Mat &output);
void thrmag(cv::Mat &input,int thr, cv::Mat &output);

//Hough
Mat HoughSpace( Mat src);
//Mat ellips(Mat src);

//COnectric circles
int selectCircles(vector<Vec3f> circles, vector<Vec3f>& concentricCircles, Point& targetCenter);
int concentricCircleDet(Mat src, Mat& dst);
void DrawRectCC(vector<Vec3f> concentricCircles, Mat dst);
//F1
//void CalculateF1(Vector<Rect>);

/** @function main */
int main( int argc, const char** argv )
{
	Mat target = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	
	//Carry out Viola Jones result
	Mat VJdst = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// Load the Strong Classifier in a structure called `cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };	
	vector<Rect> VJrect = ViolaJones(VJdst);
	
	// Show V - J result
	//imwrite("ViolaJones_Result.jpg",VJdst);
	//imshow("ViolaJones_Result",VJdst);

	//Calculate Gradient Magnitude
	Mat GradMagThr = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	GradMagThr = GradientMag(GradMagThr);
	//imshow("magnitudethreshold",GradMagThr);
	//imwrite("magnitudethreshold.jpg",GradMagThr);

	//find Houghspace of Lines, gradient magnitude image as input
	Mat houghspace = HoughSpace(GradMagThr);
	//imshow("HoughSpace",houghspace);
	//imwrite("HoughSpace.jpg",houghspace);
	
	
	//Find concentric circles
	Mat concircle;
	//vector<Vec3f> concentricCircles;
	concentricCircleDet(target, concircle);	
	for(int j=0;j< concentricCircles.size();j++)
		cout<<concentricCircles[j]<<endl;
	//imshow("concircle",concircle);

	//Draw Rect from concentric circle
	Mat concirDet = concircle;
	DrawRectCC(concentricCircles,concirDet);
	imshow("detected",concirDet);
	imwrite("detected.jpg",concirDet);
	
	
	//pickout square from ViolaJones
	Mat target_grey,target_grey_blur,target_grey_blur_Canny;
	target =   imread(argv[1], CV_LOAD_IMAGE_COLOR);

	cvtColor( target, target_grey, CV_BGR2GRAY );
	Mat pickoutVJ =  target_grey;
	for(int i = 0; i < pickoutVJ.rows;i++){
		for(int j=0;j<pickoutVJ.cols;j++){
			int flagin =0;
			for(int rectnumber = 0;  rectnumber < VJrect.size();rectnumber++){
				if(i>=VJrect[rectnumber].y && (i <= VJrect[rectnumber].y + VJrect[rectnumber].height) && 
				   j>=VJrect[rectnumber].x && (j <= VJrect[rectnumber].x + VJrect[rectnumber].width)){

					flagin = 1;
					break;
				}
				else{
					flagin = 0;
					
				}
			}
			
			if(flagin == 0)
				pickoutVJ.at<uchar>(i,j) = 0;
		}
		
	}
	//imshow("pickoutVJ",pickoutVJ);
	//imwrite("pickoutVJ.jpg",pickoutVJ);
	


	//DetectLines
	
	GaussianBlur( pickoutVJ,target_grey_blur, Size(7,7),3,2 );	
	Canny(  target_grey_blur, target_grey_blur_Canny, 40, 120, 3 );

	vector<Vec2f> lines;
	HoughLines(target_grey_blur_Canny,lines,1,CV_PI/180,60,0,0);
	Mat linesvote = cvCreateImage(cvSize(GradMagThr.cols,GradMagThr.rows), IPL_DEPTH_8U,1);
	linesvote.setTo(0);

	//Mat drawlines = GradMagThr;
	for( size_t i = 0; i < lines.size(); i++ )
	{
		Mat newline = cvCreateImage(cvSize(target.cols,target.rows), IPL_DEPTH_8U,1);
		newline.setTo(0);
		//imshow("newline",newline);
		float rho = lines[i][0], theta = lines[i][1];
	  	Point pt1, pt2;
	  	double a = cos(theta), b = sin(theta);
	 	double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
	 	pt2.y = cvRound(y0 - 1000*(a));
		line( newline, pt1, pt2, cvScalar(1), 1, CV_AA);
		linesvote = linesvote + newline;
	}	
	//imshow("linesvote",linesvote);
	
	//find brightest pixel
	Point maxpoint;
	int maxvote = 0;
	for(int i=0;i<linesvote.rows;i++){
		for(int j=0;j<linesvote.cols;j++){
			if(linesvote.at<uchar>(i,j) > maxvote){
				maxpoint = Point(j,i);
				maxvote = linesvote.at<uchar>(i,j);
			}
		}
	}
	//cout<<"maxpoint "<< maxpoint <<endl;
	Mat linescross = target;
	circle(linescross, maxpoint, 5, Scalar(0, 255, 255), -1);
	//imshow("linescross",linescross);

	
	//draw rectangle through centre
	Rect selectedRect;
	int Maxheight=0;
	Mat VJ_centre =linescross ;
	for(int i=0;i < VJrect.size();i++){
		if(maxpoint.x >=VJrect[i].y && (maxpoint.x <= VJrect[i].y + VJrect[i].height) && 
		   maxpoint.y  >=VJrect[i].x && (maxpoint.y  <= VJrect[i].x + VJrect[i].width) && VJrect[i].height > Maxheight){
			Maxheight = VJrect[i].height;
			selectedRect = VJrect[i];
		}
	}	
	
	rectangle(VJ_centre  , Point(selectedRect.x, selectedRect.y), Point(selectedRect.x + selectedRect.width, selectedRect.y + selectedRect.height), Scalar( 0, 255, 0 ), 2);
	//imshow("VJ_centre",VJ_centre );
	//imwrite("VJ_centre.jpg",VJ_centre );



	
	
	

	

	waitKey(0);
	return 0;
}
/** end of @function main */



vector<Rect> ViolaJones( Mat frame){
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	//cout << faces.size() << endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
	return faces;
}


Mat GradientMag(Mat src){

	//sobel factors	
	Mat kernel_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
 	Mat kernel_y = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

	//grey image
	Mat src_grey;
	cvtColor( src, src_grey, CV_BGR2GRAY );

	//Gaussian Blur
	Mat src_grey_blur;
	GaussianBlur( src_grey,src_grey_blur, Size(3,3),3,2 );	
	//imshow( "src_grey_blur", src_grey_blur);
	//apply sobel
	Mat x_direction, y_direction;
	conv(src_grey_blur, kernel_x, 3, x_direction);
 	conv(src_grey_blur, kernel_y, 3, y_direction);
	
	//magnitude
	Mat Mag;
	magnitude(x_direction,y_direction,Mag);
	
	//Apply threshold
	Mat MagThreshold;
	thrmag(Mag,MAGTHR,MagThreshold);
	
	return MagThreshold;

}

Mat HoughSpace( Mat src){
	double MinTheta = 0;
	double MaxTheta = 2*CV_PI;
	int houghwidth = 360;
	double anglestep = CV_PI/houghwidth;
	//cout<<anglestep<<endl;
	int houghheight = cvRound( (src.rows + src.cols+1)*2);
	
	
 	Mat houghspace = cvCreateImage(cvSize(houghwidth,houghheight), IPL_DEPTH_8U,1);
	//cout<<houghheight<<endl;
	houghspace.setTo(0);

	//imshow("black",houghspace);
	for(int i =0; i < src.rows; i++){
		for(int j=0;j < src.cols;j++){
			if(src.at<uchar>(i,j) !=0){
				double theta = 0;
				int indextheta = 0;
				while(theta < MaxTheta && indextheta < houghwidth){
					double rho =  cvRound(cos(theta)*j + sin(theta)*i) ;	
					if(rho >= 0 && rho < houghheight){
						//cout<< " i " <<i<<" j " <<j<<endl;
						//cout<<"height"<<houghspace.cols<<endl;
						//cout<<setprecision(9)<<theta<<endl;
						//cout<<"index "<<index<< " rho "<< rho << " theta "<< theta<<endl;	
						int indexrho = cvRound((houghheight/2) + rho); // start from middle
						houghspace.at<uchar>(indexrho,indextheta) = houghspace.at<uchar>(indexrho,indextheta)+1;
						//houghspace.at<uchar>(rho,indextheta) = houghspace.at<uchar>(rho,indextheta)+1;		
						indextheta++;
						theta = theta + anglestep;
						
					}
					else if(rho<0 && abs(rho)<houghheight){
						int indexrho = cvRound((houghheight/2) + rho); // start from middle
						houghspace.at<uchar>(indexrho,indextheta) = houghspace.at<uchar>(indexrho,indextheta)+1;	
						indextheta++;
						theta = theta + anglestep;
					}
					else 
						break;
				}	
			}		
		}
	}
	
	return houghspace;
	
}

void DrawLines(vector<Vec2f> lines, Mat src){
	//find piont with max value in houghspace
	/*int maxvalue = 0;
	for(int i =0;i<houghspace.rows;i++){
		for(int j=0;j<houghspace.cols;j++){
			int value = houghspace.at<uchar>(i,j);
			if (value > maxvalue)
				maxvalue = value;
				//cout<<maxvalue<<endl;
		}
	}

	int threshold = cvRound(0.6*maxvalue);
	//cout<<threshold<<endl;;

	//find lines and store in rho and theta in vector
	vector<Vec2f> lines;
	for(int i=0;i<houghspace.rows;i++){
		for(int j=0;j<houghspace.cols;j++){
			int value = houghspace.at<uchar>(i,j);
			if(houghspace.at<uchar>(i,j)>threshold)
				lines.push_back( Vec2f(i,j) );
		}
	}*/
		
	//drawlines
	for( size_t i = 0; i < lines.size(); i++ )
	{
		float rho = lines[i][0], theta = lines[i][1];
		//rho = rho - (houghspace.rows/2);

		//cout<<" rho " <<rho<<" theta "<<theta<<endl;
	  	Point pt1, pt2;
	  	double a = cos(theta), b = sin(theta);
	 	double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
	 	pt2.y = cvRound(y0 - 1000*(a));
		line( src, pt1, pt2, cvScalar(255), 1, CV_AA);
	}
	//return lines;
}



void conv(cv::Mat &input, cv::Mat &kernel, int size, cv::Mat &blurredOutput)
{
	// intialise the output using the input
	blurredOutput.create(input.size(), input.type());
	//make edge
	int kernelRadiusX = (kernel.size[0] - 1) / 2;
	int kernelRadiusY = (kernel.size[1] - 1) / 2;
	cv::Mat paddedInput;
	cv::copyMakeBorder(input, paddedInput,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv::BORDER_REPLICATE);
	// now we can do the convoltion
	for (int i = 0; i < input.rows; i++)
	{
		for (int j = 0; j < input.cols; j++)
		{
			double sum = 0.0;
			for (int m = -kernelRadiusX; m <= kernelRadiusX; m++)
			{
				for (int n = -kernelRadiusY; n <= kernelRadiusY; n++)
				{
					// find the correct indices we are using
					int imagex = i + m + kernelRadiusX;
					int imagey = j + n + kernelRadiusY;
					int kernelx = m + kernelRadiusX;
					int kernely = n + kernelRadiusY;

					// get the values from the padded image and the kernel
					int imageval = (int)paddedInput.at<uchar>(imagex, imagey);
					double kernalval = kernel.at<double>(kernelx, kernely);

					// do the multiplication
					sum += imageval * kernalval;
				}
			}
			// set the output value as the sum of the convolution
			if (sum < 0) sum = 0;
			if (sum > 255) sum = 255;
			blurredOutput.at<uchar>(i, j) = (uchar)sum;
		}
	}
}

void magnitude(cv::Mat &x, cv::Mat &y, cv::Mat &output){
	output.create(x.size(),x.type());
	for(int i=0 ; i < x.rows ; i++ ){
		for(int j=0 ; j<x.cols ; j++){
			int gx = x.at<uchar>(i,j);
			int gy = y.at<uchar>(i,j);
			int sum = sqrt(gx*gx+gy*gy);
			sum = sum > 255 ? 255 : sum;
			sum = sum < 0 ? 0 : sum;
			output.at<uchar>(i,j) = sum;
		}
		
	}
}

void thrmag(cv::Mat &input,int thr, cv::Mat &output){
	output.create(input.size(),input.type());
	for(int i=0 ; i < input.rows ; i++ )
		for(int j=0 ; j<input.cols ; j++)	
			output.at<uchar>(i,j) = input.at<uchar>(i,j) >= thr ? 255:0;

}



//Find cicles whose centres are near to each other
int selectCircles(vector<Vec3f> circles, vector<Vec3f>& concentricCircles, Point& targetCenter){
	concentricCircles.clear();
	//step1:Find point nearest to centre
	float centerDist = FLT_MAX;
	Point center;
	for (uint i = 0; i < circles.size() - 1; i++)
	{
		for (uint j = i + 1; j < circles.size(); j++)
		{
			float dist = norm(Point(circles[i][0], circles[i][1]) - Point(circles[j][0], circles[j][1]));
			if (dist < centerDist)
			{
				centerDist = dist;
				center.x = (circles[i][0] + circles[j][0]) / 2;
				center.y = (circles[i][1] + circles[j][1]) / 2;
			}
		}
	}
	//step2:filter unwanted circle
	for (uint i = 0; i < circles.size(); i++)
	{
		if (norm(Point(circles[i][0], circles[i][1]) - center) < MIN_CENTER_DIST)
		{
			concentricCircles.push_back(circles[i]);
		}
	}
	targetCenter = center;
	return 1;
}

int concentricCircleDet(Mat src, Mat& dst){
	Mat gray;
	if (src.empty())
		return 0;
	else if (src.channels() > 1)
		cvtColor(src, gray, CV_BGR2GRAY);
	else
		src.copyTo(gray);
	src.copyTo(dst);

	Mat edgeImg;
	Canny(gray, edgeImg, 100, 200, 3);

	//imshow("edge", edgeImg);

	const uint maxRadius = gray.rows/3;
	const uint minRadius = 5;
	uint step = 5;
	vector<Vec3f> circles;
	vector<Vec3f> circlesFinal;
	for (uint i = minRadius; i < maxRadius; i += step)
	{
		HoughCircles(edgeImg, circles, CV_HOUGH_GRADIENT, 3, 10, 200, 50, i, i+step);
		if (circles.empty())
			continue;
		else
		{
			circlesFinal.push_back(circles[0]);
			//Point center(cvRound(circles[0][0]), cvRound(circles[0][1]));
			//int radius = cvRound(circles[0][2]);
			//circle(dst, center, radius, Scalar(0, 0, 255), 2, 8, 0);
			circles.clear();
		}
	}
	//vector<Vec3f> concentricCircles;
	Point targetCenter2;
	selectCircles(circlesFinal, concentricCircles, targetCenter2);
	Point targetCenter = Point(0, 0);
	for (uint i = 0; i < concentricCircles.size(); i++)
	{
		Point center(cvRound(concentricCircles[i][0]), cvRound(concentricCircles[i][1]));
		targetCenter = targetCenter + center;
		int radius = cvRound(concentricCircles[i][2]);
		circle(dst, center, radius, Scalar(0, 0, 255), 2, 8, 0);
	}


	if (concentricCircles.size() > 0)
	{
		targetCenter.x = targetCenter.x / concentricCircles.size();
		targetCenter.y = targetCenter.y / concentricCircles.size();
		circle(dst, targetCenter2, 3, Scalar(0, 255, 255), -1);
	}

	return -1;
}

void DrawRectCC(vector<Vec3f> concentricCircles, Mat dst){
	int MaxCentrex = 0;
	int MaxCentrey = 0;
	int MaxRadius = 0;

	for (int i = 0; i < concentricCircles.size(); i++)
	{
		if(cvRound(concentricCircles[i][2]) > MaxRadius){
			MaxCentrex = cvRound(concentricCircles[i][0]);
			MaxCentrey = cvRound(concentricCircles[i][1]);
			MaxRadius = cvRound(concentricCircles[i][2]);
		}
			
	}

	rectangle(dst, Point(MaxCentrex-MaxRadius, MaxCentrey-MaxRadius),Point(MaxCentrex+MaxRadius, MaxCentrey+MaxRadius), Scalar( 0, 255, 0 ), 2);
	cout<<MaxCentrex<<endl;
	cout<<MaxCentrey<<endl;
	cout<<MaxRadius<<endl;
	
}



void Loaddata(string filename){
	
	string lineA;
	ifstream fileIn;
	//open file and error check
	fileIn.open(filename.c_str());
	if (fileIn.fail()) {
		cerr << " * the file you are trying to access cannot be found or opened";
		exit(1);
	}

	//initialize vector
	vector<int> temp;
	for(int i=0;i<7;i++){temp.push_back(0);}
	
	while ( fileIn.good() ) {
		while (getline(fileIn, lineA)) {
			istringstream stream(lineA);
			stream>>temp[0]>>temp[1]>>temp[2]>>temp[3]>>temp[4]>>temp[5]>>temp[6];
			imagedata.push_back(temp);
		}
	}
}
/*void F1score(vector<Rect> detectedRec,Mat src){
	//Load Data
	Loaddata("data.txt");
	for(int i=0;i<imagedata.size();i++){
		for(int j=0;j<7;j++){
			cout<<imagedata[i][j]<<" ";
		}
		cout<<endl;
	}
	//find which image it is
	vector<int> imagenubmer;
	for(int j=0;j<imagedata.size();j++){
			if(imagedata[j][5] == src.rows && imagedata[j][6] == src.cols)
				imagenumber.push_back(j);
	}
	
	if(imagenumber.size() == 1 && detectedRec.size() == 1){
		//find centre of detected Rec
		
			
	}
	else{
		
	}


}*/

/*Mat ellips(Mat src){
	
	RNG rng(12345);
	int max_thresh = 255;
	Mat src_gray;
	Mat threshold_output;
	blur( src, src_gray, Size(3,3) );
	vector<vector<Point> > contours;
  	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold( src_gray, threshold_output, 100, 255, THRESH_BINARY );

	/// Find contours
	findContours( threshold_output, contours, hierarchy, CV_RETR_EXTERNAL, 			CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

	vector<RotatedRect> minRect( contours.size() );
	vector<RotatedRect> minEllipse( contours.size() );

	for( int i = 0; i < contours.size(); i++ ){
		 minRect[i] = minAreaRect( Mat(contours[i]) );
       		if( contours[i].size() > 5 ){
			 minEllipse[i] = fitEllipse( Mat(contours[i]) ); 
		}
    	 }

	Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
	for( int i = 0; i< contours.size(); i++ )
     	{
      		 Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
      		 
      		 ellipse( drawing, minEllipse[i], color, 2, 8 );      		 
     	}
	 //imshow( "Contours", drawing );
	return drawing;


}*/


