#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

vector<double> normalizeHistogram(Mat src) {
	vector<double> histogram(256, 0.0f);

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

Mat doubleThresholding(Mat src, int lowTh, int highTh) {
	Mat gauss, out = Mat::zeros(src.rows, src.cols, CV_8U);
	GaussianBlur(src, gauss, Size(3, 3), 0);

	for (int i = 0; i < gauss.rows; i++) {
		for (int j = 0; j < gauss.cols; j++) {
			auto pixel = gauss.at<uchar>(i, j);

			if (pixel > highTh) {
				out.at<uchar>(i, j) = 255;
			}
			else if (pixel > lowTh) {
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

	int lowTh = 0, highTh = 0;
	vector<double> prob(3, 0.0f), cumulativeMean(3, 0.0f);
	double maxVariance = 0.0f;

	for (int i = 0; i < histogram.size() - 2; i++) {
		prob[0] += histogram[i];
		cumulativeMean[0] += (i + 1) * histogram[i];
		
		for (int j = 0; j < histogram.size() - 1; j++) {
			prob[1] += histogram[j];
			cumulativeMean[1] += (j + 1) * histogram[j];
			
			for (int k = 0; k < histogram.size(); k++) {
				prob[2] += histogram[k];
				cumulativeMean[2] += (k + 1) * histogram[k];

				auto currentVariance = pow(cumulativeMean[0] / prob[0] - globalMean, 2) + pow(cumulativeMean[1] / prob[1] - globalMean, 2) + pow(cumulativeMean[2] / prob[2] - globalMean, 2);
				if (currentVariance > maxVariance) {
					maxVariance = currentVariance;
					highTh = i;
					lowTh = j;
				}
			}
		}
	}

	return doubleThresholding(src, lowTh, highTh);
}

int main() {
	Mat img = imread("../images/galaxies.tif", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error reading image";
		return -1;
	}

	imshow("original", img);
	imshow("otsu-2 image", otsu2(img));
	waitKey(0);

	return 0;
}