#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

vector<double> normalizeHistogram(Mat src) {
	vector<double> histogram(256, 0);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			histogram[src.at<uchar>(i, j)]++;
		}
	}

	for (int i = 0; i < histogram.size(); i++) {
		histogram[i] /= src.rows * src.cols;
	}

	return histogram;
}

Mat doubleThresholding(Mat src, int threshLow, int threshHigh) {
	Mat out = Mat::zeros(src.rows, src.cols, CV_8U);

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) > threshHigh) {
				out.at<uchar>(i, j) = 255;
			}
			else if (src.at<uchar>(i, j) > threshLow) {
				out.at<uchar>(i, j) = 127;
			}
		}
	}

	return out;
}

Mat otsu2(Mat src) {
	vector<double> histogram = normalizeHistogram(src);

	double globalMean = 0.0f;
	for (int i = 0; i < histogram.size(); i++) {
		globalMean += (i + 1) * histogram[i];
	}

	vector<double> prob(3, 0);
	vector<double> cumulativeMean(3, 0);
	vector<int> thresh(2, 0);
	double currentVariance, maxVariance = 0.0f;

	for (int i = 0; i < histogram.size() - 2; i++) {
		prob[0] += histogram[i];
		cumulativeMean[0] += (i + 1) * histogram[i];

		for (int j = i + 1; j < histogram.size() - 1; j++) {
			prob[1] += histogram[j];
			cumulativeMean[1] += (j + 1) * histogram[j];

			for (int l = j + 1; l < histogram.size(); l++) {
				prob[2] += histogram[l];
				cumulativeMean[2] += (l + 1) * histogram[l];

				currentVariance = 
					prob[0] * pow( cumulativeMean[0] / prob[0] - globalMean, 2)
					+ prob[1] * pow(cumulativeMean[1] / prob[1] - globalMean, 2) 
					+ prob[2] * pow(cumulativeMean[2] / prob[2] - globalMean, 2);

				if (currentVariance > maxVariance) {
					maxVariance = currentVariance;
					thresh[0] = i;
					thresh[1] = j;
				}
			}
			prob[2] = 0; cumulativeMean[2] = 0;
		}
		prob[1] = 0; cumulativeMean[1] = 0;
	}

	Mat gauss;
	GaussianBlur(src, gauss, Size(3, 3), 0, 0);

	Mat out = doubleThresholding(gauss, thresh[0], thresh[1]);

	return out;
}

int main() {
	Mat img = imread("../images/galaxies.tif", IMREAD_GRAYSCALE);
	
	if (img.empty()) {
		cout << "Error reading image" << endl;
		exit(-1);
	}

	Mat otsu2Img = otsu2(img);

	imshow("original", img);
	imshow("otsu K2", otsu2Img);

	waitKey(0);

	return 0;
}