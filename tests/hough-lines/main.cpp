#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat drawLines(Mat src, int distance, Mat votes, int votesTh) {
	Mat out = src.clone();

	for (int i = 0; i < votes.rows; i++) {
		for (int j = 0; j < votes.cols; j++) {
			if (votes.at<uchar>(i, j) > votesTh) {
				double theta = (j - 90) * CV_PI / 180;
				double cos_t = cos(theta), sin_t = sin(theta);
				int x = (i - distance) * cos_t, y = (i - distance) * sin_t;
				Point pt1(cvRound(x + distance * (-sin_t)), cvRound(y + distance * cos_t));
				Point pt2(cvRound(x - distance * (-sin_t)), cvRound(y - distance * cos_t));

				line(out, pt1, pt2, Scalar(127), 2);
			}
		}
	}

	return out;
}

Mat houghLines(Mat src, int lowTh, int highTh, int votesTh) {
	int distance = hypot(src.rows, src.cols);
	Mat gauss, edges, votes = Mat::zeros(distance * 2, 180, CV_8U);
	GaussianBlur(src, gauss, Size(3, 3), 0);
	Canny(gauss, edges, lowTh, highTh);

	for (int i = 0; i < edges.rows; i++) {
		for (int j = 0; j < edges.cols; j++) {
			if (edges.at<uchar>(i, j) == 255) {
				for (int theta = 0; theta < 180; theta++) {
					auto rho = distance + j * cos((theta - 90) * CV_PI / 180) + i * sin((theta - 90) * CV_PI / 180);
					votes.at<uchar>(rho, theta)++;
				}
			}
		}
	}

	return drawLines(src, distance, votes, votesTh);
}

int main() {
	Mat img = imread("../images/jetplane.tif", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);
	imshow("hough lines image", houghLines(img, 120, 160, 103));
	waitKey(0);

	return 0;
}