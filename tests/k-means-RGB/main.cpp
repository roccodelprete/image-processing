#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

Mat kMeans(Mat src, int k, double th) {
	Mat out = src.clone();
	vector<Vec3d> oldMean(k, 0), newMean(k, 0);
	vector<vector<Point>> clusters(k);
	vector<Vec3b> centers(k, 0);
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
				double minDistance = DBL_MAX;
				int minDistanceIndex = 0;

				for (int i = 0; i < centers.size(); i++) {
					auto pixel = src.at<Vec3b>(x, y);
					double distanceBlue = centers[i].val[0] - pixel.val[0];
					double distanceGreen = centers[i].val[1] - pixel.val[1];
					double distanceRed = centers[i].val[2] - pixel.val[2];
					double distance = sqrt(pow(distanceBlue, 2) + pow(distanceGreen, 2) + pow(distanceRed, 2));

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
				newMean[i].val[0] += src.at<Vec3b>(clusters[i][j].x, clusters[i][j].y).val[0];
				newMean[i].val[1] += src.at<Vec3b>(clusters[i][j].x, clusters[i][j].y).val[1];
				newMean[i].val[2] += src.at<Vec3b>(clusters[i][j].x, clusters[i][j].y).val[2];
			}

			newMean[i].val[0] /= clusters[i].size();
			newMean[i].val[1] /= clusters[i].size();
			newMean[i].val[2] /= clusters[i].size();
			centers[i] = Vec3b(newMean[i]);
		}

		for (int i = 0; i < newMean.size(); i++) {
			double distanceBlue = newMean[i].val[0] - oldMean[i].val[0];
			double distanceGreen = newMean[i].val[1] - oldMean[i].val[1];
			double distanceRed = newMean[i].val[2] - oldMean[i].val[2];
			double distance = sqrt(pow(distanceBlue, 2) + pow(distanceGreen, 2) + pow(distanceRed, 2));

			if (distance > th) {
				isChanged = false;
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
	srand(time(NULL));

	Mat img = imread("../images/lenna.jpg", IMREAD_COLOR);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	imshow("original", img);
	imshow("k-means image", kMeans(img, 8, .01));
	waitKey(0);

	return 0;
}