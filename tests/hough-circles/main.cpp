#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat drawCircles(Mat src, Mat edges, Mat votes, int votesTh, int minRadius, int maxRadius) {
	Mat out = src.clone();

	for (int radius = minRadius; radius <= maxRadius; radius++) {
		for (int b = 0; b < edges.rows; b++) {
			for (int a = 0; a < edges.cols; a++) {
				if (votes.at<uchar>(b, a, radius - minRadius) > votesTh) {
					circle(out, Point(a, b), 2, Scalar(0), 2);
					circle(out, Point(a, b), radius, Scalar(0), 2);
				}
			}
		}
	}

	return out;
}

Mat houghCircles(Mat src, int minRadius, int maxRadius, int lowTh, int highTh, int votesTh) {
	int sizes[3] = { src.rows, src.cols, maxRadius - minRadius + 1 };
	Mat gauss, edges, votes = Mat::zeros(3, sizes, CV_8U);
	GaussianBlur(src, gauss, Size(5, 5), 0);
	Canny(gauss, edges, lowTh, highTh);

	for (int i = 0; i < edges.rows; i++) {
		for (int j = 0; j < edges.cols; j++) {
			if (edges.at<uchar>(i, j) == 255) {
				for (int radius = minRadius; radius <= maxRadius; radius++) {
					for (int theta = 0; theta < 360; theta++) {
						int a = j - radius * cos(theta * CV_PI / 180), b = i - radius * sin(theta * CV_PI / 180);

						if (a >= 0 && b >= 0 && a < edges.cols && b < edges.rows) {
							votes.at<uchar>(b, a, radius - minRadius)++;
						}
					}
				}
			}
		}
	}

	return drawCircles(src, edges, votes, votesTh, minRadius, maxRadius);
}

int main() {
	Mat img = imread("../images/coins.png", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);
	imshow("hough circles image", houghCircles(img, 40, 90, 160, 230, 140));
	waitKey();

	return 0;
}