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

Mat otsu(Mat src) {
	vector<double> histogram = normalizeHistogram(src);

	double globalMean = 0.0f;
	for (int i = 0; i < histogram.size(); i++) {
		globalMean += (i + 1) * histogram[i];
	}

	int th = 0;
	double prob = 0.0f, cumulativeMean = 0.0f, maxVariance = 0.0f;

	for (int i = 0; i < histogram.size(); i++) {
		prob += histogram[i];
		cumulativeMean += (i + 1) * histogram[i];
		double currentVariance = pow(prob * globalMean - cumulativeMean, 2) / (prob * (1 - prob));

		if (currentVariance > maxVariance) {
			th = i;
			maxVariance = currentVariance;
		}
	}

	Mat gauss, out;
	GaussianBlur(src, gauss, Size(3, 3), 0);
	threshold(gauss, out, th, 255, THRESH_BINARY);

	return out;
}

int main() {
	Mat img = imread("../images/galaxies.tif", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);
	imshow("otsu image", otsu(img));
	waitKey(0);

	return 0;
}