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
	stack<Point> stack;
	stack.push(seed);

	while (!stack.empty()) {
		Point center = stack.top();
		mask.at<uchar>(center) = 1;
		stack.pop();

		for (int i = 0; i < 8; i++) {
			Point estimatedPoint = center + points[i];

			if (estimatedPoint.x >= 0 && estimatedPoint.y >= 0 && estimatedPoint.x < src.cols && estimatedPoint.y < src.rows) {
				Vec3b srcCenterPixel = src.at<Vec3b>(center), srcEstimatedPointPixel = src.at<Vec3b>(estimatedPoint);
				int distance = pow(srcCenterPixel[0] - srcEstimatedPointPixel[0], 2) + pow(srcCenterPixel[1] - srcEstimatedPointPixel[1], 2) + pow(srcCenterPixel[2] - srcEstimatedPointPixel[2], 2);

				if (distance < th && mask.at<uchar>(estimatedPoint) == 0 && out.at<uchar>(estimatedPoint) == 0) {
					stack.push(estimatedPoint);
					mask.at<uchar>(estimatedPoint) = 1;
				}
			}
		}
	}

	return mask;
}

int regionGrowingRGB(Mat src, double minRegionFactor, int maxRegionNumber, int th) {
	int minRegionArea = int(minRegionFactor * src.rows * src.cols);
	Mat out = Mat::zeros(src.rows, src.cols, CV_8U), mask = Mat::zeros(src.rows, src.cols, CV_8U);
	uchar labels = 1;

	for (int x = 0; x < src.cols; x++) {
		for (int y = 0; y < src.rows; y++) {
			if (out.at<uchar>(Point(x, y)) == 0) {
				mask = grow(src, out, mask, th, Point(x, y));
				int maskArea = int(sum(mask)[0]);

				if (maskArea > minRegionArea) {
					out += mask * labels;

					imshow("region", mask * 255);
					waitKey();

					labels++;

					if (labels > maxRegionNumber) {
						cout << "Over-segmentation" << endl;
						return -2;
					}
				}
				else {
					out += mask * 255;
				}
			}

			mask -= mask;
		}
	}

	return 0;
}

int main() {
	Mat img = imread("../images/lenna.jpg", IMREAD_COLOR);

	if (img.empty()) {
		cout << "Error reading images" << endl;
		return -1;
	}

	if (img.rows > 500 || img.cols > 500) {
		resize(img, img, Size(0, 0), .5, .5);
	}

	imshow("original", img);

	return regionGrowingRGB(img, .04, 5, 15);
}