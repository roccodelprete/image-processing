#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

vector<double> normalizeHistogram(Mat src) {
	vector<double> histogram(256, 0.0);

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
	double globalMean = 0.0, prob = 0.0, cumulativeMean = 0.0, maxVariance = 0.0;
	int th = 0;
	Mat gauss, out;

	for (int i = 0; i < histogram.size(); i++) {
		globalMean += (i + 1) * histogram[i];
	}

	for (int i = 0; i < histogram.size(); i++) {
		prob += histogram[i];
		cumulativeMean += (i + 1) * histogram[i];
		double variance = pow(prob * globalMean - cumulativeMean, 2) / (prob * (1 - prob));

		if (variance > maxVariance) {
			maxVariance = variance;
			th = i;
		}
	}

	GaussianBlur(src, gauss, Size(5, 5), 0);
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
	waitKey();

	return 0;
}