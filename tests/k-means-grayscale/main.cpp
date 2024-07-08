#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <vector>

using namespace std;
using namespace cv;

Mat kMeans(Mat src, int k, double th) {
	Mat out = src.clone();
	vector<double> newMean(k, 0.0), oldMean(k, 0.0);
	vector<vector<Point>> clusters(k);
	vector<uchar> centers(k, 0);
	bool isChanged = true;

	for (int i = 0; i < centers.size(); i++) {
		centers[i] = src.at<uchar>(rand() % src.rows, rand() % src.cols);
	}

	while (isChanged) {
		isChanged = false;

		for (int i = 0; i < clusters.size(); i++) {
			oldMean[i] = newMean[i];
			newMean[i] = 0;
			clusters[i].clear();
		}

		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				int minDistance = INT_MAX, minDistanceIndex = 0;

				for (int t = 0; t < centers.size(); t++) {
					int distance = abs(src.at<uchar>(i, j) - centers[t]);

					if (distance < minDistance) {
						minDistance = distance;
						minDistanceIndex = t;
					}
				}

				clusters[minDistanceIndex].push_back(Point(i, j));
			}
		}

		for (int i = 0; i < clusters.size(); i++) {
			for (int j = 0; j < clusters[i].size(); j++) {
				newMean[i] += src.at<uchar>(clusters[i][j].x, clusters[i][j].y);
			}

			newMean[i] /= clusters[i].size();
			centers[i] = uchar(newMean[i]);
		}

		for (int i = 0; i < k; i++) {
			if (abs(newMean[i] - oldMean[i]) > th) {
				isChanged = true;
				break;
			}
		}
	}

	for (int i = 0; i < clusters.size(); i++) {
		for (int j = 0; j < clusters[i].size(); j++) {
			out.at<uchar>(clusters[i][j].x, clusters[i][j].y) = centers[i];
		}
	}

	return out;
}

int main() {
	srand(time(NULL));

	Mat img = imread("../images/mountains.png", IMREAD_GRAYSCALE);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);
	imshow("k-means image", kMeans(img, 3, .01));
	waitKey(0);

	return 0;
}