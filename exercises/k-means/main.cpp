#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat kMeansGrayScale(Mat src, int k, double thresh) {
	srand(time(NULL));

	vector<uchar>centers(k, 0);

	for (int i = 0; i < k; i++) {
		int randomRow = rand() % src.rows;
		int randomCol = rand() % src.cols;

		centers[i] = src.at<uchar>(randomRow, randomCol);
	}

	vector<double> oldMean(k, 0);
	vector<double> newMean(k, 0);
	vector<vector<Point>> cluster(k);
	bool isChanged = true;

	while (isChanged) {
		isChanged = false;

		for (int i = 0; i < k; i++) {
			oldMean[i] = newMean[i];
			newMean[i] = 0;
			cluster[i].clear();
		}

		for (int x = 0; x < src.rows; x++) {
			for (int y = 0; y < src.cols; y++) {
				int minDistance = INT_MAX, minDistanceIndex = 0;

				for (int i = 0; i < cluster.size(); i++) {
					int distance = abs(centers[i] - src.at<uchar>(x, y));

					if (distance < minDistance) {
						minDistance = distance;
						minDistanceIndex = i;
					}
				}

				cluster[minDistanceIndex].push_back(Point(x, y));
			}
		}

		for (int i = 0; i < cluster.size(); i++) {
			for (int j = 0; j < cluster[i].size(); j++) {
				newMean[i] += src.at<uchar>(cluster[i][j].x, cluster[i][j].y);
			}

			newMean[i] /= cluster[i].size();
			centers[i] = uchar(newMean[i]);
		}

		for (int i = 0; i < k; i++) {
			if (abs(newMean[i] - oldMean[i]) > thresh) {
				isChanged = true;
				break;
			}
		}
	}

	Mat out = src.clone();
	for (int i = 0; i < cluster.size(); i++) {
		for (int j = 0; j < cluster[i].size(); j++) {
			out.at<uchar>(cluster[i][j].x, cluster[i][j].y) = centers[i];
		}
	}

	return out;
}

Mat kMeansRGB(Mat src, int k, double thresh) {
	srand(time(NULL));

	vector<Vec3b>centers(k, 0);

	for (int i = 0; i < k; i++) {
		int randomRow = rand() % src.rows;
		int randomCol = rand() % src.cols;

		centers[i] = src.at<uchar>(randomRow, randomCol);
	}

	vector<Vec3d> oldMean(k, 0);
	vector<Vec3d> newMean(k, 0);
	vector<vector<Point>> cluster(k);
	bool isChanged = true;

	while (isChanged) {
		isChanged = false;

		for (int i = 0; i < k; i++) {
			oldMean[i] = newMean[i];
			newMean[i] = 0;
			cluster[i].clear();
		}

		for (int x = 0; x < src.rows; x++) {
			for (int y = 0; y < src.cols; y++) {
				double minDistance = DBL_MAX;
				int minDistanceIndex = 0;

				for (int i = 0; i < cluster.size(); i++) {
					double distanceRed = centers[i].val[0] - src.at<Vec3b>(x, y).val[0];
					double distanceBlue = centers[i].val[1] - src.at<Vec3b>(x, y).val[1];
					double distanceGreen = centers[i].val[2] - src.at<Vec3b>(x, y).val[2];

					double distance = sqrt(pow(distanceRed, 2) + pow(distanceBlue, 2) + pow(distanceGreen, 2));

					if (distance < minDistance) {
						minDistance = distance;
						minDistanceIndex = i;
					}
				}

				cluster[minDistanceIndex].push_back(Point(x, y));
			}
		}

		for (int i = 0; i < cluster.size(); i++) {
			for (int j = 0; j < cluster[i].size(); j++) {
				newMean[i].val[0] += src.at<Vec3b>(cluster[i][j].x, cluster[i][j].y).val[0];
				newMean[i].val[1] += src.at<Vec3b>(cluster[i][j].x, cluster[i][j].y).val[1];
				newMean[i].val[2] += src.at<Vec3b>(cluster[i][j].x, cluster[i][j].y).val[2];
			}

			newMean[i].val[0] /= cluster[i].size();
			newMean[i].val[1] /= cluster[i].size();
			newMean[i].val[2] /= cluster[i].size();

			centers[i] = Vec3b(newMean[i]);
		}

		for (int i = 0; i < k; i++) {
			double distanceRed = newMean[i].val[0] - oldMean[i].val[0];
			double distanceBlue = newMean[i].val[1] - oldMean[i].val[1];
			double distanceGreen = newMean[i].val[2] - oldMean[i].val[2];

			double distance = sqrt(pow(distanceRed, 2) + pow(distanceBlue, 2) + pow(distanceGreen, 2));

			if (abs(distance) > thresh) {
				isChanged = true;
				break;
			}
		}
	}

	Mat out = src.clone();
	for (int i = 0; i < cluster.size(); i++) {
		for (int j = 0; j < cluster[i].size(); j++) {
			out.at<Vec3b>(cluster[i][j].x, cluster[i][j].y) = centers[i];
		}
	}

	return out;
}

int main() {
	Mat img = imread("../images/mountains.png", IMREAD_GRAYSCALE);
	Mat imgRGB = imread("../images/lenna.jpg", IMREAD_COLOR);

	if (img.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}
	
	if (imgRGB.empty()) {
		cout << "Error reading image" << endl;
		return -1;
	}

	Mat kMeansGrayScaleImg = kMeansGrayScale(img, 3, .01);
	Mat kMeansRGBImg = kMeansRGB(imgRGB, 8, .01);

	imshow("original", img);
	imshow("k-means", kMeansGrayScaleImg);

	imshow("original RGB", imgRGB);
	imshow("k-means RGB", kMeansRGBImg);

	waitKey(0);

	return 0;
}