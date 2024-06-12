#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat checkVotesAndDrawLine(Mat src, Mat votes, int distance, int voteThresholding) {
	Mat out = src.clone();

	for (int i = 0; i < votes.rows; i++) {
		for (int j = 0; j < votes.cols; j++) {
			if (votes.at<uchar>(i, j) > voteThresholding) {
				double theta = (j - 90) * CV_PI / 180;
				double cos_t = cos(theta);
				double sin_t = sin(theta);
				int x = (i - distance) * cos_t;
				int y = (i - distance) * sin_t;

				Point pt1(cvRound(x + distance * (-sin_t)), cvRound(y + distance * cos_t));
				Point pt2(cvRound(x - distance * (-sin_t)), cvRound(y - distance * cos_t));

				line(out, pt1, pt2, Scalar(127), 2);
			}
		}
	}

	return out;
}

Mat houghLines(Mat src, int lowThresholding, int highThresholding, int voteThresholding) {
	int distance = hypot(src.rows, src.cols);
	Mat votes = Mat::zeros(distance * 2, 180, CV_8U);

	Mat gauss, edges;
	GaussianBlur(src, gauss, Size(5, 5), 0, 0);
	Canny(gauss, edges, lowThresholding, highThresholding);

	double rho, theta;

	for (int x = 0; x < edges.rows; x++) {
		for (int y = 0; y < edges.cols; y++) {
			if (edges.at<uchar>(x, y) == 255) {
				for (theta = 0; theta < 180; theta++) {
					rho = distance + (y * cos((theta - 90) * CV_PI / 180)) + (x * sin((theta - 90) * CV_PI / 180));
					votes.at<uchar>(rho, theta)++;
				}
			}
		}
	}

	Mat out = checkVotesAndDrawLine(src, votes, distance, voteThresholding);

	return out;
}

Mat checkVotesAndDrawCircle(Mat src, Mat edges, Mat votes, int minRadius, int maxRadius, int voteThresholding) {
	Mat out = src.clone();

	for (int radius = minRadius; radius <= maxRadius; radius++) {
		for (int b = 0; b < edges.rows; b++) {
			for (int a = 0; a < edges.cols; a++) {
				if (votes.at<uchar>(b, a, radius - minRadius) > voteThresholding) {
					circle(out, Point(a, b), 3, Scalar(0), 2);
					circle(out, Point(a, b), radius, Scalar(0), 2);
				}
			}
		}
	}

	return out;
}

Mat houghCircle(Mat src, int lowThresholding, int highThresholding, int voteThresholding, int minRadius, int maxRadius) {
	int sizes[] = { src.rows, src.cols, maxRadius - minRadius + 1 };

	Mat votes = Mat::zeros(3, sizes, CV_8U);

	Mat gauss, edges;
	GaussianBlur(src, gauss, Size(7, 7), 0, 0);
	Canny(gauss, edges, lowThresholding, highThresholding);

	for (int x = 0; x < edges.rows; x++) {
		for (int y = 0; y < edges.cols; y++) {
			if (edges.at<uchar>(x, y) == 255) {
				for (int radius = minRadius; radius <= maxRadius; radius++) {
					for (int theta = 0; theta < 360; theta++) {
						int a = y - radius * cos(theta * CV_PI / 180);
						int b = x - radius * sin(theta * CV_PI / 180);

						if (a >= 0 && a < edges.cols && b >= 0 && b < edges.rows) {
							votes.at<uchar>(b, a, radius - minRadius)++;
						}
					}
				}
			}
		}
	}

	Mat out = checkVotesAndDrawCircle(src, edges, votes, minRadius, maxRadius, voteThresholding);

	return out;
}

int main() {
	Mat imgForHoughLines = imread("../images/jetplane.tif", IMREAD_GRAYSCALE);
	Mat imgForHoughCircles = imread("../images/coins.png", IMREAD_GRAYSCALE);

	if (imgForHoughLines.empty()) {
		cout << "Error reading image" << endl;
		exit(-1);
	}
	
	if (imgForHoughCircles.empty()) {
		cout << "Error reading image" << endl;
		exit(-1);
	}

	Mat houghLinesImg = houghLines(imgForHoughLines, 90, 160, 100);
	Mat houghCirclesImg = houghCircle(imgForHoughCircles, 150, 230, 140, 40, 90);

	imshow("original", imgForHoughLines);
	imshow("hough lines", houghLinesImg);
	imshow("hough circles", houghCirclesImg);

	waitKey(0);

	return 0;
}