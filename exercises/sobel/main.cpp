#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat sobel(Mat src, int depth, int xOrder, int yOrder) {
	float sobelMatX[] = {
		-1.0, -2.0, -1.0,
		0.0, 0.0, 0.0,
		1.0, 2.0, 1.0
	};
	
	float sobelMatY[] = {
			-1.0, 0.0, 1.0,
			-2.0, 0.0, 2.0,
			-1.0, 0.0, 1.0
	};

	Mat sobelMaskX = Mat(3, 3, CV_32F, sobelMatX);
	Mat sobelMaskY = Mat(3, 3, CV_32F, sobelMatY);

	Mat out;

	if (xOrder == 1 && yOrder == 0) {
		filter2D(src, out, depth, sobelMaskX);
	}
	else if (xOrder == 0 && yOrder == 1) {
		filter2D(src, out, depth, sobelMaskY);
	}
	else {
		Mat sobelFilteredX = sobel(src, depth, 1, 0);
		Mat sobelFilteredY = sobel(src, depth, 0, 1);

		out = abs(sobelFilteredX) + abs(sobelFilteredY);
	}

	return out;
}

int main() {
	Mat img = imread("../../images/cameraman.tif", IMREAD_GRAYSCALE);

	int xOrder, yOrder;

	cout << "Insert x order: ";
	cin >> xOrder;
	
	cout << "\n" << "Insert y order: ";
	cin >> yOrder;

	if (xOrder > 1 && xOrder < 0) {
		cout << "Insert right x order!" << "\n";

		exit(-1);
	}
	
	if (yOrder > 1 && yOrder < 0) {
		cout << "Insert right y order!" << "\n";

		exit(-1);
	}

	Mat sobelImg = sobel(img, CV_32F, xOrder, yOrder);
	
	Mat defaultSobelImg;
	Sobel(img, defaultSobelImg, CV_32F, xOrder, yOrder);

	imshow("original image", img);
	imshow("sobel image with my sobel", sobelImg);
	imshow("sobel image with default sobel", defaultSobelImg);

	waitKey(0);

	return 0;
}