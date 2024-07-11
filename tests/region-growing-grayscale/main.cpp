#include <iostream>
#include <opencv2/opencv.hpp>
#include <stack>

using namespace std;
using namespace cv;

Mat grow(Mat src, Mat out, Mat mask, int th, Point seed) {
	Point points[] = {
		Point(-1, -1), Point(-1, 0), Point(-1, 1),
		Point(0, -1), Point(0, 1),
		Point(1, -1), Point(1, 0), Point(1, 1),
	};
	stack<Point> pointsStack;
	pointsStack.push(seed);

	while (!pointsStack.empty()) {
		Point center = pointsStack.top();
		mask.at<uchar>(center) = 1;
		pointsStack.pop();

		for (int i = 0; i < 8; i++) {
			Point estimatedPoint = center + points[i];

			if (estimatedPoint.x >= 0 && estimatedPoint.y >= 0 && estimatedPoint.y < src.rows && estimatedPoint.x < src.cols) {
				uchar delta = abs(src.at<uchar>(center) - src.at<uchar>(estimatedPoint));

				if (delta < th && mask.at<uchar>(estimatedPoint) == 0 && out.at<uchar>(estimatedPoint) == 0) {
					mask.at<uchar>(estimatedPoint) = 1;
					pointsStack.push(estimatedPoint);
				}
			}
		}
	}

	return mask;
}

int regionGrowingGrayscale(Mat src, int th, double minRegionFactor, int maxRegionNumber) {
	int minRegionArea = int(src.rows * src.cols * minRegionFactor);
	Mat out = Mat::zeros(src.rows, src.cols, CV_8U), mask = Mat::zeros(src.rows, src.cols, CV_8U);
	uchar labels = 1;

	for (int i = 0; i < src.cols; i++) {
		for (int j = 0; j < src.rows; j++) {
			if (out.at<uchar>(i, j) == 0) {
				mask = grow(src, out, mask, th, Point(i, j));
				int maskArea = int(sum(mask).val[0]);

				if (maskArea > minRegionArea) {
					out += mask * labels;

					imshow("region", mask * 255);
					waitKey();
					labels++;

					if (labels > maxRegionNumber) {
						cout << "oversegmentation" << endl;
						return -2;
					}
				}
			}
			else {
				out += mask * 255;
			}
				
			mask -= mask;
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

	return regionGrowingGrayscale(img, 3, .01, 5);
}