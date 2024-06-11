#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

/**
* This function applies the Laplacian of Gaussian to the source image
* The Laplacian of Gaussian marks image borders
* @param src Source image
* @param depth Depth of the output image
* @param ksize Size of filter
* @returns The image with LoG applied
*/
Mat LoG(Mat src, int depth, int ksize) {
	Mat output, gaussianImg;

	GaussianBlur(src, gaussianImg, Size(ksize, ksize), 0, 0);
	Laplacian(gaussianImg, output, depth, ksize);

	return output;
}

/**
* Function to detect zero-crossing points on an image.
* Zero-crossing detects points where there is a change in pixel intensity
* from positive to negative or vice versa.
* @param src The source image (expected to be a single-channel, floating-point image)
* @param depth Depth of the output image
* @param ksize Size of filter
* @return The image with zero-crossing detected, where zero-crossing points are marked in white.
*/
Mat zeroCrossing(Mat src, int ksize) {
	Mat LoGImg = LoG(src, CV_32F, ksize);

	Mat out = Mat::zeros(src.size(), CV_8U);
	
	double srcMinValue, srcMaxValue;
	minMaxLoc(LoGImg, &srcMinValue, &srcMaxValue);

	double threshold = srcMaxValue * .0;

	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			// Matrix around with current i and j (inverted because i is y axis and j is x axis
			Mat around = LoGImg(Rect(j - 1, i - 1, 3, 3));

			double aroundMinValue, aroundMaxValue;
			minMaxLoc(around, &aroundMinValue, &aroundMaxValue);

			/**
			* condition to detect if there is a change from 1 to 0 or 0 to 1
			* in change from 1 to 0: if flag is true, there is a zero-crossing in the change from 1 to 0; false otherwise
			* in change from 0 to 1: if flag is true, there is a zero-crossing in the change from 0 to 1; false otherwise
			*/
			bool zeroCrossingFlag = LoGImg.at<float>(i, j) > 0 ? (aroundMinValue) < 0 : (aroundMaxValue) > 0;

			if (zeroCrossingFlag && (abs(aroundMaxValue - aroundMinValue)) > threshold) {
				out.at<uchar>(i, j) = 255;
			}
		}
	}

	return out;
}

int main() {
	Mat img = imread("../images/cameraman.tif", IMREAD_GRAYSCALE);
	Mat zeroCrossingImg = zeroCrossing(img, 1);

	imshow("original", img);
	imshow("zero-crossing", zeroCrossingImg);

	waitKey(0);

	return 0;
}