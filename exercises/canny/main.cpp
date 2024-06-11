#include <opencv2/opencv.hpp>

using namespace cv;

/**
* Implementa l'algoritmo di Canny
*/

Mat nonMaximaSuppression(Mat magnitude, Mat orientations) {
	Mat out = Mat::zeros(magnitude.rows, magnitude.cols, CV_8UC1);
	float angle;

	for (int i = 1; i < magnitude.rows - 1; i++) {
		for (int j = 1; j < magnitude.cols - 1; j++) {
			angle = orientations.at<float>(i, j) > 180 ? orientations.at<float>(i, j) - 360 : orientations.at<float>(i, j);

			if ((angle > -22.5 && angle <= 22.5) || (angle > 157.5 || angle <= -157.5)) { // horizontal edge
				if (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i, j - 1) && magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i, j + 1)) {
					out.at<uchar>(i, j) = magnitude.at<uchar>(i, j);
				}
			}
			else if ((angle > -67.5 && angle <= -22.5) || (angle > 112.5 && angle <= 157.5)) { // +45° edge
				if (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i + 1, j - 1) && magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i - 1, j + 1)) {
					out.at<uchar>(i, j) = magnitude.at<uchar>(i, j);
				}
			}
			else if ((angle > -112.5 && angle <= -67.5) || (angle > 67.5 && angle <= 112.5)) { // vertical edge
				if (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i - 1, j) && magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i + 1, j)) {
					out.at<uchar>(i, j) = magnitude.at<uchar>(i, j);
				}
			}
			else if ((angle > -157.5 && angle <= -112.5) || (angle <= 67.5 && angle > 22.5)) { // -45° edge
				if (magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i - 1, j - 1) && magnitude.at<uchar>(i, j) > magnitude.at<uchar>(i + 1, j + 1)) {
					out.at<uchar>(i, j) = magnitude.at<uchar>(i, j);
				}
			}
		}
	}

	return out;
}

Mat hysterisisThresholding(Mat nms, double lowThresholding, double highThresholding) {
	Mat out = Mat::zeros(nms.rows, nms.cols, CV_8UC1);

	for (int i = 1; i < nms.rows - 1; i++) {
		for (int j = 1; j < nms.cols - 1; j++) {
			if (nms.at<uchar>(i, j) > highThresholding) {
				out.at<uchar>(i, j) = 255;
				
				for (int k = -1; k <= 1; k++) {
					for (int t = -1; t <= 1; t++) {
						if (nms.at<uchar>(i + k, j + t) > lowThresholding && nms.at<uchar>(i + k, j + t) < highThresholding) {
							out.at<uchar>(i + k, j + t) = 255;
						}
					}
				}
			}
		}
	}

	return out;
}

Mat canny(Mat src, double lowThresholding, double highThresholding, int ksize) {
	Mat gaussianBlurImg;
	GaussianBlur(src, gaussianBlurImg, Size(3, 3), 0, 0);

	Mat dx, dy, magnitude, orientations, dx2, dy2;
	Sobel(gaussianBlurImg, dx, CV_32FC1, 1, 0, ksize);
	Sobel(gaussianBlurImg, dy, CV_32FC1, 0, 1, ksize);
	pow(dx, 2, dx2);
	pow(dy, 2, dy2);
	sqrt(dx2 + dy2, magnitude);
	normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8UC1);
	phase(dx, dy, orientations, true);

	Mat nms = nonMaximaSuppression(magnitude, orientations);

	return hysterisisThresholding(nms, lowThresholding, highThresholding);
}

int main() {
	Mat img = imread("../images/cameraman.tif", IMREAD_GRAYSCALE);

	Mat cannyImg = canny(img, 40, 100, 3);

	imshow("original", img);
	imshow("canny", cannyImg);

	waitKey(0);

	return 0;
}