#include <iostream>
#include <opencv2/opencv.hpp>
#include <stack>

using namespace std;
using namespace cv;

Mat grow(Mat src, Mat mask, Mat out, Point seed, int th) {
	Point pointsShift[8] = {
		Point(-1, -1), Point(-1, 0), Point(-1, 1),
		Point(0, -1), Point(0, 1),
		Point(1, -1), Point(1, 0), Point(1, 1)
	};

	stack<Point> pointsStack;
	pointsStack.push(seed);

	while (!pointsStack.empty()) {
		auto center = pointsStack.top();
		mask.at<uchar>(center) = 1;
		pointsStack.pop();

		for (int i = 0; i < 8; i++) {
			auto estimatedPoint = center + pointsShift[i];

			if (estimatedPoint.x >= 0 && estimatedPoint.y >= 0 && estimatedPoint.x < src.cols && estimatedPoint.y < src.rows) {
				auto srcEstimatedPoint = src.at<uchar>(estimatedPoint), srcCenterPoint = src.at<uchar>(center);
				int delta = abs(srcCenterPoint - srcEstimatedPoint);

				if (delta < th && out.at<uchar>(estimatedPoint) == 0 && mask.at<uchar>(estimatedPoint) == 0) {
					mask.at<uchar>(estimatedPoint) = 1;
					pointsStack.push(estimatedPoint);
				}
			}
		}
	}

	return mask;
}

int regionGrowingGrayscale(Mat src, int th, int maxRegionNumber, double minRegionFactor = .01) {
	Mat mask = Mat::zeros(src.rows, src.cols, CV_8U), out = Mat::zeros(src.rows, src.cols, CV_8U);
	uchar labels = 1;
	int minRegionArea = int(src.rows * src.cols * minRegionFactor);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (out.at<uchar>(i, j) == 0) {
				mask = grow(src, mask, out, Point(j, i), th);
				int maskArea = countNonZero(mask);

				if (maskArea > minRegionArea) {
					out.setTo(labels, mask);

					imshow("region", mask * 255);
					waitKey(0);

					labels++;

					if (labels > maxRegionNumber) {
						cout << "Oversegmentation" << endl;
						return -2;
					}
				}

				mask.setTo(0);
			}
		}
	}

	return 0;
}

int main() {
	Mat img = imread("../images/lenna.jpg", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);
	
	return regionGrowingGrayscale(img, 3, 10);
}