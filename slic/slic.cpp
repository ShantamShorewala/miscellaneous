#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

using namespace std;

int main(){

	int k;
	cout << "Enter the number of superpixels (k) \n";
	cin >> k;

	cv::Mat image = cv::imread("1.jpg");
	cv::resize(image, image, cv::Size(image.cols/4,image.rows/4));
	cv::cvtColor(image, image, cv::COLOR_BGR2Lab);
	float N = image.rows * image.cols;

	for (int i=0; i<50; i++){
		cv::imshow("image", image);
		cv::waitKey(10);
	}

	float S = sqrt(N/k);

	vector<vector<float>> centers;
	float x = 10, y =10;
	cv::Vec3f pixelValue;

	for (int i=0; i<k; i++){
		pixelValue = image.at<cv::Vec3b>(x,y);
		centers.push_back(vector<float>{x, y, pixelValue.val[0], pixelValue.val[1], pixelValue.val[2]});
		x += S;
		if (x>image.rows){
			y += S;
			x = 10;
		}
	}

	cv::Mat sobelX, sobelY;
	cv::Sobel(image, sobelX, CV_64F, 1, 0, 3);
	cv::Sobel(image, sobelY, CV_64F, 0, 1, 3);
	sobelX += sobelY;

	cout << centers.size() << "\n";
	vector<float> idx;
	for (int i=0; i < centers.size(); i++){
		x = centers[i][0];
		y = centers[i][1];
		float min = sobelX.at<float>(x,y);
		for (int j=-1; j < 2; j++){
			for (int k=-1; k < 2; k++){
				if (sobelX.at<float>(x+j,y+k) < min){
					min = sobelX.at<float>(x+j,y+k);
					idx.push_back(x+j);
					idx.push_back(y+k); 
				}
			}
		}
		pixelValue = image.at<cv::Vec3b>(idx[0],idx[1]);
		centers[i] = vector<float>{idx[0], idx[1], pixelValue.val[0], pixelValue.val[1], pixelValue.val[2]};
	} 

	cout << "Completed perturbing";
}