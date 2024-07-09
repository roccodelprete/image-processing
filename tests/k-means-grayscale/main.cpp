#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <vector>

using namespace std;
using namespace cv;

Mat kMeans(Mat src, int k, double th) {
	Mat out = src.clone();
	vector<double> oldMean(k, 0.0), newMean(k, 0.0);
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

		for (int x = 0; x < src.rows; x++) {
			for (int y = 0; y < src.cols; y++) {
				int minDistance = INT_MAX, minDistanceIndex = 0;

				for (int i = 0; i < centers.size(); i++) {
					int distance = abs(src.at<uchar>(x, y) - centers[i]);

					if (distance < minDistance) {
						minDistance = distance;
						minDistanceIndex = i;
					}
				}

				clusters[minDistanceIndex].push_back(Point(x, y));
			}
		}

		for (int i = 0; i < clusters.size(); i++) {
			for (int j = 0; j < clusters[i].size(); j++) {
				newMean[i] += src.at<uchar>(clusters[i][j].x, clusters[i][j].y);
			}

			newMean[i] /= clusters[i].size();
			centers[i] = uchar(newMean[i]);
		}

		for (int i = 0; i < newMean.size(); i++) {
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