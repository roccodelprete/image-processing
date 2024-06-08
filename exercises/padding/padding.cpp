#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

static unsigned char avgValue3x3(Mat image, int rowStart, int colStart) {
	int sum = 0;

	for (int i = rowStart - 1; i < rowStart + 2; i++) {
		for (int j = colStart - 1; j < colStart + 2; j++) {
			sum += image.at<unsigned char>(i, j);
		}
	}

	return (unsigned char)(sum / 9);
}

Vec3b avgValue3x3Colorized(Mat image, int rowStart, int colStart) {
	Vec3i sum(0, 0, 0);

	for (int i = rowStart; i < rowStart + 2; i++) {
		for (int j = colStart; j < colStart + 2; j++) {
			sum += image.at<Vec3b>(i, j);
		}
	}

	return Vec3b(sum / 9);
}

Mat greysAvg(Mat img) {
	Mat paddedImage;

	Scalar value(0);    // black border
	copyMakeBorder(img, paddedImage, 1, 1, 1, 1, BORDER_CONSTANT, value);
	Mat result(img.rows, img.cols, img.type());

	for (int i = 1; i < paddedImage.rows - 1; i++) {
		for (int j = 1; j < paddedImage.cols - 1; j++) {
			result.at<unsigned char>(i - 1, j - 1) = avgValue3x3(paddedImage, i, j);
		}
	}

	return result;
}

Mat colorizedAvg(Mat img) {
	Mat paddedImage;

	Scalar value(0, 0, 0);    // black border
	copyMakeBorder(img, paddedImage, 1, 1, 1, 1, BORDER_CONSTANT, value);
	Mat result(img.rows, img.cols, img.type());

	for (int i = 1; i < paddedImage.rows - 1; i++) {
		for (int j = 1; j < paddedImage.cols - 1; j++) {
			result.at<Vec3b>(i - 1, j - 1) = avgValue3x3Colorized(paddedImage, i, j);
		}
	}

	return result;
}

void showImage(string windowName, Mat image, bool waitkey) {
	namedWindow(windowName, WINDOW_AUTOSIZE);
	imshow(windowName, image);

	if (waitkey) {
		waitKey(0);
	}
}

int main()
{
	Mat image = imread("building.png", -1);
	Mat secondImage = imread("lenna.jpg", -1);

	showImage("building", image, false);
	showImage("lena", secondImage, false);

	Mat paddedImage = greysAvg(image);
	Mat colorizedPaddedImage = colorizedAvg(secondImage);

	showImage("building padded", paddedImage, false);
	showImage("lena padded", colorizedPaddedImage, true);

	return 0;
}
