#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

Mat kMeansRGB(Mat src, int k, double th) {
	vector<vector<Point>> clusters(k);
	vector<Vec3b> centers(k, Vec3b(0, 0, 0));
	vector<Vec3d> newMean(k, Vec3d(0.0, 0.0, 0.0)), oldMean(k, Vec3d(0.0, 0.0, 0.0));
	bool isChanged = true;
	Mat out = src.clone();

	for (int i = 0; i < centers.size(); i++) {
		centers[i] = src.at<Vec3b>(rand() % src.rows, rand() % src.cols);
	}

	while (isChanged) {
		isChanged = false;

		for (int i = 0; i < k; i++) {
			oldMean[i] = newMean[i];
			newMean[i] = Vec3d(0.0, 0.0, 0.0);
			clusters[i].clear();
		}

		for (int x = 0; x < src.rows; x++) {
			for (int y = 0; y < src.cols; y++) {
				double minDistance = DBL_MAX;
				int minDistanceIndex = 0;

				for (int i = 0; i < centers.size(); i++) {
					Vec3b center = centers[i], point = src.at<Vec3b>(x, y);
					double distance = sqrt(pow(center[0] - point[0], 2) + pow(center[1] - point[1], 2) + pow(center[2] - point[2], 2));

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
				Vec3b point = src.at<Vec3b>(clusters[i][j].x, clusters[i][j].y);
				newMean[i][0] += point[0];
				newMean[i][1] += point[1];
				newMean[i][2] += point[2];
			}

			newMean[i][0] /= clusters[i].size();
			newMean[i][1] /= clusters[i].size();
			newMean[i][2] /= clusters[i].size();

			centers[i] = Vec3b(newMean[i]);
		}

		for (int i = 0; i < newMean.size(); i++) {
			double distance = sqrt(pow(newMean[i][0] - oldMean[i][0], 2) + pow(newMean[i][1] - oldMean[i][1], 2) + pow(newMean[i][2] - oldMean[i][2], 2));

			if (distance > th) {
				isChanged = true;
				break;
			}
		}
	}

	for (int i = 0; i < clusters.size(); i++) {
		for (int j = 0; j < clusters[i].size(); j++) {
			out.at<Vec3b>(clusters[i][j].x, clusters[i][j].y) = centers[i];
		}
	}

	return out;
}

int main() {
	Mat img = imread("../images/lenna.jpg", IMREAD_COLOR);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);
	imshow("k-means RGB image", kMeansRGB(img, 5, .01));
	waitKey();

	return 0;
}