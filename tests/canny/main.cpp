#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std; using namespace cv;

Mat nonMaximaSuppression(Mat magnitude, Mat orientations) {
	Mat out = Mat::zeros(magnitude.rows, magnitude.cols, CV_8UC1);
	for (int i = 1; i < magnitude.rows - 1; i++) {
		for (int j = 1; j < magnitude.cols - 1; j++) {
			float angle = orientations.at<float>(i, j) > 180 ? orientations.at<float>(i, j) - 360 : orientations.at<float>(i, j);
			auto magnitudePoint = magnitude.at<uchar>(i, j);
			if ((angle > -22.5 && angle <= 22.5) || (angle > 157.5 || angle <= -157.5)) {
				if (magnitudePoint > magnitude.at<uchar>(i, j - 1) && magnitudePoint > magnitude.at<uchar>(i, j + 1)) {
					out.at<uchar>(i, j) = magnitudePoint;
				}
			}
			else if ((angle > -67.5 && angle <= -22.5) || (angle > 112.5 && angle <= 157.5)) {
				if (magnitudePoint > magnitude.at<uchar>(i + 1, j - 1) && magnitudePoint > magnitude.at<uchar>(i - 1, j + 1)) {
					out.at<uchar>(i, j) = magnitudePoint;
				}
			}
			else if ((angle > -112.5 && angle <= -67.5) || (angle > 67.5 && angle <= 112.5)) {
				if (magnitudePoint > magnitude.at<uchar>(i + 1,j) && magnitudePoint > magnitude.at<uchar>(i - 1, j)) {
					out.at<uchar>(i, j) = magnitudePoint;
				}
			}
			else if ((angle > -157.5 && angle <= -112.5) || (angle > 22.5 && angle <= 67.5)) {
				if (magnitudePoint > magnitude.at<uchar>(i + 1, j + 1) && magnitudePoint > magnitude.at<uchar>(i - 1, j - 1)) {
					out.at<uchar>(i, j) = magnitudePoint;
				}
			}
		}
	}

	return out;
}

Mat hysterisisTh(Mat nms, int lowTh, int highTh) {
	Mat out = Mat::zeros(nms.rows, nms.cols, CV_8UC1);
	for (int i = 1; i < nms.rows - 1; i++) {
		for (int j = 1; j < nms.cols - 1; j++) {
			if (nms.at<uchar>(i, j) > highTh) {
				out.at<uchar>(i, j) = 255;
				for (int k = -1; k < 2; k++) {
					for (int t = -1; t < 2; t++) {
						if (nms.at<uchar>(i + k, j + t) > lowTh && nms.at<uchar>(i + k, j + t) < highTh) {
							out.at<uchar>(i + k, j + t) = 255;
						}
					}
				}
			}
		}
	}

	return out;
}

Mat canny(Mat src, int lowTh, int highTh, int ksize = 3) {
	Mat gauss, dx, dy, dx2, dy2, magnitude, orientations;
	GaussianBlur(src, gauss, Size(3, 3), 0);
	Sobel(gauss, dx, CV_32FC1, 1, 0, ksize);
	Sobel(gauss, dy, CV_32FC1, 0, 1, ksize);
	pow(dx, 2, dx2);
	pow(dy, 2, dy2);
	sqrt(dx2 + dy2, magnitude);
	normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8UC1);
	phase(dx, dy, orientations, true);
	Mat nms = nonMaximaSuppression(magnitude, orientations);
	return hysterisisTh(nms, lowTh, highTh);
}

int main() {
	Mat img = imread("../images/cameraman.tif", IMREAD_GRAYSCALE);
	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	Mat cannyImg = canny(img, 40, 120);

	imshow("original", img);
	imshow("canny image", cannyImg);

	waitKey(0);

	return 0;
} 