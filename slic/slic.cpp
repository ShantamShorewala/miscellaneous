#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>

using namespace std;

#define MAX_ITERS 10

float computeDistance(cv::Mat image, float x, float y, float j, float k, float S, int m){

	cv::Vec3f pixelValue, pixelValueCenter;
	pixelValue = image.at<cv::Vec3b>(x+j,y+k);
	pixelValueCenter = image.at<cv::Vec3b>(x,y);

	float dc = sqrt(pow(pixelValue.val[0] - pixelValueCenter.val[0],2) + pow(pixelValue.val[1] - pixelValueCenter.val[1],2) + pow(pixelValue.val[2] - pixelValueCenter.val[2],2));
	float ds = sqrt(pow(j,2)+pow(k,2));

	float dist = sqrt(pow(dc,2) + pow(ds/S, 2)*pow(m,2));
	return dist;

}

vector<vector<int>> connectedComponents(vector<vector<int>> labels, int n){

	vector<int> dx = {-1,0,1,0};
	vector<int> dy = {0,-1,0,1};
	// int SuperSize = (labels.size() * labels[0].size())/n;
	int adjLabel = 0;

	vector<int> xvec((labels.size()*labels[0].size()));
	vector<int> yvec((labels.size()*labels[0].size()));

	// cout << labels.size() << "  " << labels[0].size() << "\n"; 
	vector<vector<int>> newLabels(labels.size(), vector<int>(labels[0].size(), -1));
	// return labels;
	
	int x1, y1, count;

	for(int x=0; x<labels.size(); x++){
		for(int y=0; y<labels[0].size(); y++){

			if (newLabels[x][y]==-1){
				newLabels[x][y] = labels[x][y];
				xvec[0] = x;
				yvec[0] = y;

				for (int i=0; i<dx.size(); i++){
					x1 = x + dx[i];
					y1 = y + dy[i];
					if (x1>=0 && x1<labels.size() && y1>=0 && y1<labels[0].size()){
						if (newLabels[x][y]>=0) adjLabel = newLabels[x][y];
					}
				}

				count=0;
				for (int i=0; i<count; i++){
					for(int j=0; j<dx.size(); j++){
						x1 = x + dx[j];
						y1 = y + dy[j];
						if (x1 >=0 && x1 < labels.size() && y1 >= 0 && y1 < labels[0].size()){
							if (newLabels[x][y]<0 && labels[x][y]==labels[x1][y1]){
								xvec[count] = x1;
								yvec[count] = y1;
								newLabels[x1][y1] = newLabels[x][y];
								count++;
							}
						}
					}
				}
 			}
		}
	}

	cout << "Completed enforcing connectivity \n";

	return newLabels;
}

int main(){

	int k, m;
	cout << "Enter the number of superpixels (k) \n";
	cin >> k;
	cout << "Enter the compactness (m) \n";
	cin >> m;

	cv::Mat image = cv::imread("1.jpg");
	cv::resize(image, image, cv::Size(image.cols/8,image.rows/8));
	cv::Mat original = image.clone();
	cv::cvtColor(image, image, cv::COLOR_BGR2Lab);
	float N = image.rows * image.cols;

	cout << "rows " << image.rows << "\n";
	cout << "columns " << image.cols << "\n";

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
	vector<float> idx = vector<float>{0,0};
	
	for (int i=0; i < centers.size(); i++){
		// cout << i << "\n";
		x = centers[i][0];
		y = centers[i][1];
		idx.clear();
		float min = sobelX.at<float>(x,y);
		for (int j=-1; j < 2; j++){
			for (int k=-1; k < 2; k++){
				if (sobelX.at<float>(x+j,y+k) < min){
					min = sobelX.at<float>(x+j,y+k);
					idx[0] = (x+j);
					idx[1] = (y+k); 
				}
			}
		}
		pixelValue = image.at<cv::Vec3b>(idx[0],idx[1]);
		centers[i] = vector<float>{idx[0], idx[1], pixelValue.val[0], pixelValue.val[1], pixelValue.val[2]};
	} 

	cout << "Completed perturbing \n";

	vector<vector<int>> labels(image.rows, vector<int>(image.cols, -1));
	vector<vector<float>> distances(image.rows, vector<float>(image.cols, FLT_MAX));
	map<int, vector<int>> cluster_centers;
	map<int, vector<vector<int>>> clusters;
	map<int, vector<vector<int>>>::iterator it;
	float d;
	vector<int> centers_count;

	for (int iter=0; iter<MAX_ITERS; iter++){

		for (int i=0; i < centers.size(); i++){
		
			x = centers[i][0];
			y = centers[i][1];
			// cout << x << " " << y << "\n";

			for (int j=-1*S; j<S; j++){
				for (int k=-1*S; k<S; k++){
					if ((x+j)>0 && (x+j)<image.rows && (y+k)>0 && (y+k)<image.cols){
						// cout << "Going well " << x+j << " " << y+k << "\n"; 
						d = computeDistance(image,x,y,j,k,S,m);
						if (d < distances[x+j][y+k]){
							distances[x+j][y+k] = d;
							labels[x+j][y+k] = i;
						}
					}
				}
			}
		}

		for (int i=0; i < centers.size(); i++)
			centers[i] = vector<float>{0.,0.,0.,0.,0.};

		centers_count = vector<int>(centers.size(), 0);

		for (int i=0; i<image.rows; i++){
			for (int j=0; j<image.cols; j++){
				if (labels[i][j]!=-1){
					pixelValue = image.at<cv::Vec3b>(i,j);
					centers[labels[i][j]][0] += i;
					centers[labels[i][j]][1] += j;
					centers[labels[i][j]][2] += pixelValue[0];
					centers[labels[i][j]][3] += pixelValue[1];
					centers[labels[i][j]][4] += pixelValue[2];
					centers_count[labels[i][j]] +=1;
				}
			}
		}

		for (int i=0; i < centers.size(); i++){
			centers[i][0] /= centers_count[i];
			centers[i][1] /= centers_count[i];
			centers[i][2] /= centers_count[i];
			centers[i][3] /= centers_count[i];
			centers[i][4] /= centers_count[i];
		}

	}

	cout << "Completed finding the cluster centers \n";

	labels = connectedComponents(labels, S);

	vector<vector<int>> temp;
	for (int i=0; i<image.rows; i++){
		for (int j=0; j<image.cols; j++){

			temp = clusters[labels[i][j]];
			// temp.clear();
			temp.push_back(vector<int>{i,j});
			clusters[labels[i][j]] =  temp;
		}
	}

	cout << "Completed labelling \n";

	cv::Mat mask, edges, gray, masked_img;
    vector<vector<cv::Point> > contours;
    vector<cv::Vec4i> hierarchy;

	for (int i=0; i<centers.size(); i++){
		
		mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC3);
		temp = clusters[i];
		for (auto pt : temp){
			x = pt[0];
			y = pt[1];
			mask.at<cv::Vec3b>(x,y)[0] = 255;
			mask.at<cv::Vec3b>(x,y)[1] = 255;
			mask.at<cv::Vec3b>(x,y)[2] = 255;
		}

		cv::cvtColor(mask, gray, cv::COLOR_BGR2GRAY);
		// cv::bitwise_and(gray, cv::Scalar(255),  masked_img, mask);
		cv::Canny(gray, edges, 100, 200, 3);
		cv::findContours(gray, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

	 //    cv::Mat drawing = cv::Mat::zeros(edges.size(), CV_8UC3);
		float maxSize = 0, id = 0;
	    for(size_t i = 0; i<contours.size(); i++)
	    {
	    	if (cv::contourArea(contours[i])>maxSize){
	    		maxSize = cv::contourArea(contours[i]);
	    		id = i;
	    	}
	    }

        cv::Scalar color = cv::Scalar(0,0,255);
        cv::drawContours(original, contours, id, color, 2, cv::LINE_8, hierarchy, 0);

		// for (int i=0; i<100; i++){
		// 	cv::imshow("image", original);
		// 	cv::waitKey(10);
		// }
	}

	cv::imwrite("connectedSegmented.png", original);

}