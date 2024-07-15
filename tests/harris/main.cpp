#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat drawCircles(Mat R, int th) {
	Mat out;
	convertScaleAbs(R, out);

	for (int i = 0; i < R.rows; i++) {
		for (int j = 0; j < R.cols; j++) {
			if (int(R.at<float>(i, j)) > th) {
				circle(out, Point(j, i), 3, Scalar(0), 2);
			}
		}
	}

	return out;
}

Mat harris(Mat src, int th, double k) {
	Mat dx, dy, dxy, dx2, dy2, c_00, c_11, c_10, trace2, mainDiagonal, secondaryDiagonal;

	Sobel(src, dx, CV_32FC1, 1, 0);
	Sobel(src, dy, CV_32FC1, 0, 1);
	multiply(dx, dy, dxy);
	pow(dx, 2, dx2);
	pow(dy, 2, dy2);
	GaussianBlur(dx2, c_00, Size(7, 7), 0);
	GaussianBlur(dy2, c_11, Size(7, 7), 0);
	GaussianBlur(dxy, c_10, Size(7, 7), 0);
	Mat c_01 = c_10.clone();
	multiply(c_11, c_00, mainDiagonal);
	multiply(c_10, c_01, secondaryDiagonal);

	Mat det = mainDiagonal - secondaryDiagonal;
	Mat trace = c_00 + c_11;
	pow(trace, 2, trace2);
	Mat R = det - k * trace2;
	normalize(R, R, 0, 255, NORM_MINMAX);

	return drawCircles(R, th);
}

int main() {
	Mat img = imread("../images/coins.png", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);
	imshow("harris image", harris(img, 120, .01));
	waitKey();

	return 0;
}