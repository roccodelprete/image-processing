#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat drawCircles(Mat src, Mat edges, Mat votes, int minRadius, int maxRadius, int votesTh) {
	Mat out = src.clone();

	for (int radius = minRadius; radius <= maxRadius; radius++) {
		for (int i = 0; i < edges.cols; i++) {
			for (int j = 0; j < edges.rows; j++) {
				if (votes.at<uchar>(j, i, radius - minRadius) > votesTh) {
					circle(out, Point(i, j), 3, Scalar(0), 2);
					circle(out, Point(i, j), radius, Scalar(0), 2);
				}
			}
		}
	}

	return out;
}

Mat houghCircles(Mat src, int lowTh, int highTh, int votesTh, int minRadius, int maxRadius) {
	int sizes[3] = { src.rows, src.cols, maxRadius - minRadius + 1 };
	Mat votes = Mat::zeros(3, sizes, CV_8U), gauss, edges;
	GaussianBlur(src, gauss, Size(5, 5), 0);
	Canny(gauss, edges, lowTh, highTh);

	for (int i = 0; i < edges.rows; i++) {
		for (int j = 0; j < edges.cols; j++) {
			if (edges.at<uchar>(i, j) == 255) {
				for (int radius = minRadius; radius <= maxRadius; radius++) {
					for (int theta = 0; theta < 360; theta++) {
						int x = j - radius * sin(theta * CV_PI / 180), y = i - radius * cos(theta * CV_PI / 180);

						if (x >= 0 && y >= 0 && x < edges.cols && y < edges.rows) {
							votes.at<uchar>(y, x, radius - minRadius)++;
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
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);
	imshow("hough circles image", houghCircles(img, 160, 230, 140, 40, 90));
	waitKey(0);

	return 0;
}