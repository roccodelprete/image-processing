#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat drawCircles(Mat src, Mat edges, Mat votes, int minRadius, int maxRadius, int votesTh) {
	Mat out = src.clone();
	for (int radius = minRadius; radius <= maxRadius; radius++) {
		for (int b = 0; b < edges.rows; b++) {
			for (int a = 0; a < edges.cols; a++) {
				if (votes.at<uchar>(b, a, radius - minRadius) > votesTh) {
					circle(out, Point(a, b), 3, Scalar(0), 2);
					circle(out, Point(a, b), radius, Scalar(0), 2);
				}
			}
		}
	}

	return out;
}

Mat houghCircles(Mat src, int lowTh, int highTh, int minRadius, int maxRadius, int votesTh) {
	int sizes[] = { src.rows, src.cols, maxRadius - minRadius + 1 };
	Mat votes = Mat::zeros(3, sizes, CV_8U), gauss, edges;
	GaussianBlur(src, gauss, Size(7, 7), 0);
	Canny(gauss, edges, lowTh, highTh);
	for (int i = 0; i < edges.rows; i++) {
		for (int j = 0; j < edges.cols; j++) {
			if (edges.at<uchar>(i, j) == 255) {
				for (int radius = minRadius; radius <= maxRadius; radius++) {
					for (int theta = 0; theta < 360; theta++) {
						int a = j - radius * cos(theta * CV_PI / 180), b = i - radius * sin(theta * CV_PI / 180);
						if (a >= 0 && a < edges.cols && b >= 0 && b < edges.rows) {
							votes.at<uchar>(b, a, radius - minRadius)++;
						}
					}
				}
			}
		}
	}

	return drawCircles(src, edges, votes, minRadius, maxRadius, votesTh);
}

int main() {
	Mat img = imread("../images/coins.png", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error reading image"  << endl;
		return -1;
	}

	Mat houghImg = houghCircles(img, 150, 230, 40, 90, 140);

	imshow("original", img);
	imshow("hough image", houghImg);

	waitKey(0);

	return 0;
}