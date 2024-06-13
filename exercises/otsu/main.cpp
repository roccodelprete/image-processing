#include <opencv2/opencv.hpp>
#include <iostream>

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

	int thresh = 0;
	double currentVariance, maxVariance = 0, cumulativeMean = 0, prob = 0;

	for (int i = 0; i < histogram.size(); i++) {
		prob += histogram[i];
		cumulativeMean += (i + 1) * histogram[i];

		currentVariance = pow(prob * globalMean - cumulativeMean, 2) / (prob * (1 - prob));

		if (currentVariance > maxVariance) {
			maxVariance = currentVariance;
			thresh = i;
		}
	}

	Mat gauss, out;

	GaussianBlur(src, gauss, Size(3, 3), 0, 0);
	threshold(gauss, out, thresh, 255, THRESH_BINARY);

	return out;
}

int main() {
	Mat img = imread("../images/galaxies.tif", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		exit(-1);
	}


	Mat otsuImg = otsu(img);

	imshow("original", img);
	imshow("otsu", otsuImg);

	waitKey(0);

	return 0;
}