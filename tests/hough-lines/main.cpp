#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat drawLines(Mat src, Mat votes, int distance, int votesTh) {
	Mat out = src.clone();

	for (int i = 0; i < votes.rows; i++) {
		for (int j = 0; j < votes.cols; j++) {
			if (votes.at<uchar>(i, j) > votesTh) {
				double theta = (j - 90) * CV_PI / 180;
				int x = (i - distance) * cos(theta), y = (i - distance) * sin(theta);
				Point pt1(cvRound(x - distance * (-sin(theta))), cvRound(y - distance * cos(theta)));
				Point pt2(cvRound(x + distance * (-sin(theta))), cvRound(y + distance * cos(theta)));
				line(out, pt1, pt2, Scalar(127), 2);
			}
		}
	}

	return out;
}

Mat houghLines(Mat src, int lowTh, int highTh, int votesTh) {
	int distance = hypot(src.rows, src.cols);
	Mat gauss, edges, votes = Mat::zeros(distance * 2, 180, CV_8U);

	GaussianBlur(src, gauss, Size(5, 5), 0);
	Canny(gauss, edges, lowTh, highTh);

	for (int i = 0; i < edges.rows; i++) {
		for (int j = 0; j < edges.cols; j++) {
			if (edges.at<uchar>(i, j) == 255) {
				for (int theta = 0; theta < 180; theta++) {
					double rho = distance + j * cos((theta - 90) * CV_PI / 180) + i * sin((theta - 90) * CV_PI / 180);
					votes.at<uchar>(rho, theta)++;
				}
			}
		}
	}

	return drawLines(src, votes, distance, votesTh);
}

int main() {
	Mat img = imread("../images/jetplane.tif", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);
	imshow("hough lines image", houghLines(img, 110, 160, 100));
	waitKey();

	return 0;
}