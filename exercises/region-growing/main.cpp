#include <opencv2/opencv.hpp>
#include <iostream>
#include <stack>

using namespace std;
using namespace cv;

Mat grow(Mat src, Mat out, Mat mask, Point seed, int thresh) {
	Point pointShift[8] = {
		Point(-1, -1), Point(-1, 0), Point(-1, 1),
		Point(0, -1), Point(0, 1),
		Point(1, -1), Point(1, 0), Point(1, 1)
	};

	stack<Point> pointStack;
	pointStack.push(seed);

	while (!pointStack.empty()) {
		Point center = pointStack.top();
		mask.at<uchar>(center) = 1;
		pointStack.pop();

		for (int i = 0; i < 8; i++) {
			Point estimatedPoint = center + pointShift[i];

			if (estimatedPoint.x < 0 || estimatedPoint.y < 0 || estimatedPoint.x > src.cols - 1 || estimatedPoint.y > src.rows - 1) {
				continue;
			}
			else {
				// uchar delta = abs(src.at<uchar>(center)  - src.at<uchar>(estimatedPoint));
				/**
				*	if the image is in RGB, this is the delta calculation
				*/
				int delta = pow(src.at<Vec3b>(center)[0] - src.at<Vec3b>(estimatedPoint)[0], 2) + pow(src.at<Vec3b>(center)[1] - src.at<Vec3b>(estimatedPoint)[1], 2) + pow(src.at<Vec3b>(center)[2] - src.at<Vec3b>(estimatedPoint)[2], 2);

				if ((out.at<uchar>(estimatedPoint) == 0) && (delta < thresh) && (mask.at<uchar>(estimatedPoint) == 0)) {
					mask.at<uchar>(estimatedPoint) = 1;
					pointStack.push(estimatedPoint);
				}
			}
		}
	}

	return mask;
}

Mat regionGrowing(Mat src, double minRegionFactor, int maxRegionNumber, int thresh) {
	uchar labels = 1;
	int minRegionArea = int(minRegionFactor * src.rows * src.cols);

	Mat mask = Mat::zeros(src.rows, src.cols, CV_8U);
	Mat out = Mat::zeros(src.rows, src.cols, CV_8U);

	for (int x = 0; x < src.cols; x++) {
		for (int y = 0; y < src.rows; y++) {
			if (out.at<uchar>(Point(x, y)) == 0) {
				mask = grow(src, out, mask, Point(x, y), thresh);
				int maskArea = (int)sum(mask).val[0];
								
				if (maskArea > minRegionArea) {
					out += labels * mask;

					labels++;
					
					imshow("region", mask * 255);
					waitKey(0);

					if (labels > maxRegionNumber) {
						cout << "oversegmentation" << endl;
						exit(-2);
					}
				}
				else {
					out += mask * 255;
				}

				mask -= mask;
			}
		}
	}

	return out;
}

int main() {
	Mat img = imread("../images/lenna.jpg", IMREAD_COLOR);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		exit(-1);
	}

	if (img.rows > 500 || img.cols > 500) {
		resize(img, img, Size(0, 0), 0.5, 0.5);
	}

	imshow("original", img);

	Mat regionGrowingImg = regionGrowing(img, .01, 10, 200);

	waitKey(0);

	return 0;
}