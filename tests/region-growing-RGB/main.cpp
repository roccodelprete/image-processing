#include <iostream>
#include <opencv2/opencv.hpp>
#include <stack>

using namespace std;
using namespace cv;

Mat grow(Mat src, Mat out, Mat mask, int th, Point seed) {
	Point points[] = {
		Point(-1, -1), Point(-1, 0), Point(-1, 1),
		Point(0, -1), Point(0, 1),
		Point(1, -1), Point(1, 0), Point(1, 1)
	};
	stack<Point> stack;

	stack.push(seed);

	while (!stack.empty()) {
		Point center = stack.top();
		mask.at<uchar>(center) = 1;
		stack.pop();

		for (int i = 0; i < 8; i++) {
			Point point = center + points[i];

			if (point.x >= 0 && point.y >= 0 && point.x < src.cols && point.y < src.rows) {
				Vec3b srcCenter = src.at<Vec3b>(center), srcPoint = src.at<Vec3b>(point);
				double delta = pow(srcCenter[0] - srcPoint[0], 2) + pow(srcCenter[1] - srcPoint[1], 2) + pow(srcCenter[2] - srcPoint[2], 2);

				if (delta < th && out.at<uchar>(point) == 0 && mask.at<uchar>(point) == 0) {
					mask.at<uchar>(point) = 1;
					stack.push(point);
				}
			}
		}
	}

	return mask;
}

int regionGrowingRGB(Mat src, int th, int maxRegionNumber, double minRegionFactor) {
	int minRegionArea = int(src.rows * src.cols * minRegionFactor);
	Mat out = Mat::zeros(src.rows, src.cols, CV_8U), mask = Mat::zeros(src.rows, src.cols, CV_8U);
	uchar  labels = 1;

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
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);

	return regionGrowingRGB(img, 15, 5, .01);
}