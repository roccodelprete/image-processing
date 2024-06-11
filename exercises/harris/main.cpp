#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat circleCorners(Mat R, int thresholding) {
	Mat out;
	convertScaleAbs(R, out);

	for (int i = 0; i < R.rows; i++) {
		for (int j = 0; j < R.cols; j++) {
			if ((int)R.at<float>(i, j) > thresholding) {
				circle(out, Point(j, i), 5, Scalar(0));
			}
		}
	}

	return out;
}

Mat harris(Mat src, int thresholding, float k, int ksize = 3) {
	Mat dx, dy, dxy, dx2, dy2;

	Sobel(src, dx, CV_32FC1, 1, 0, ksize);
	Sobel(src, dy, CV_32FC1, 0, 1, ksize);

	pow(dx, 2, dx2);
	pow(dy, 2, dy2);
	multiply(dx, dy, dxy);

	Mat c_00, c_11, c_10, c_01, productPrincipalDiagonal, productSecondaryDiagonal;
	GaussianBlur(dx2, c_00, Size(7, 7), 0);
	GaussianBlur(dy2, c_11, Size(7, 7), 0);
	GaussianBlur(dxy, c_10, Size(7, 7), 0);
	c_01 = c_10;

	Mat det, trace, trace2, R;
	multiply(c_00, c_11, productPrincipalDiagonal);
	multiply(c_10, c_01, productSecondaryDiagonal);
	det = productPrincipalDiagonal - productSecondaryDiagonal;
	trace = c_00 + c_11;
	pow(trace, 2, trace2);
	R = det - (k * trace2);

	normalize(R, R, 0, 255, NORM_MINMAX, CV_32FC1);

	return circleCorners(R, thresholding);
}

int main() {
	Mat img = imread("../images/bulding.png", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error in reading image" << endl;
		exit(-1);
	}

	Mat harrisImg = harris(img, 160, 0.04);

	imshow("original", img);
	imshow("harris", harrisImg);

	waitKey(0);

	return 0;
}