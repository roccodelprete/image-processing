#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat drawLines(Mat src, Mat votes, int distance, int votesTh) {
	Mat out = src.clone();

	for (int i = 0; i < votes.rows; i++) {
		for (int j = 0; j < votes.cols; j++) {
			if (votes.at<uchar>(i, j) > votesTh) {
				double theta = (j - 90) * CV_PI / 180;
				double cos_t = cos(theta), sin_t = sin(theta);
				int x = (i - distance) * cos_t, y = (i - distance) * sin_t;
				Point pt1(cvRound(x + distance * (-sin_t)), cvRound(y + distance * cos_t)), pt2(cvRound(x - distance * (-sin_t)), cvRound(y - distance * cos_t));
				line(out, pt1, pt2, Scalar(127), 2);
			}
 		}
	}

	return out;
}

Mat houghLines(Mat src, int lowTh, int highTh, int votesTh) {
	int distance = hypot(src.rows, src.cols);
	Mat votes = Mat::zeros(distance * 2, 180, CV_8U), gauss, edges;
	GaussianBlur(src, gauss, Size(5, 5), 0);
	Canny(gauss, edges, lowTh, highTh);

	for (int x = 0; x < edges.rows; x++) {
		for (int y = 0; y < edges.cols; y++) {
			if (edges.at<uchar>(x, y) == 255) {
				for (double theta = 0; theta < 180; theta++) {
					double rho = distance + (x * sin((theta - 90) * CV_PI / 180)) + (y * cos((theta - 90) * CV_PI / 180));
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

	Mat houghLinesImg = houghLines(img, 90, 160, 100);

	imshow("original", img);
	imshow("hough image", houghLinesImg);

	waitKey(0);

	return 0;
}