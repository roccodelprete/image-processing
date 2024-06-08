#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

Mat laplaciano(Mat src, int depth, int kernelSize) {
	float dataset[9];

	switch (kernelSize) {
		case 1: {
			float dataset90[] = {
				0.0, 1.0, 0.0,
				1.0, -4.0, 1.0,
				0.0, 1.0, 0.0
			};

			copy(dataset90, dataset90 + sizeof(dataset90) / sizeof(dataset90[0]), dataset);

			break;
		}
		
		case 3: {
			float dataset45[] = {
				1.0, 1.0, 1.0,
				1.0, -8.0, 1.0,
				1.0, 1.0, 1.0
			};

			copy(dataset45, dataset45 + sizeof(dataset45) / sizeof(dataset45[0]), dataset);

			break;
		}
	}

	Mat mask = Mat(3, 3, CV_32F, dataset);
	Mat out;
	filter2D(src, out, depth, mask);

	return out;
}

int main() {
	Mat img = imread("../../images/cameraman.tif", IMREAD_GRAYSCALE);

	int filterMode;
	cout << "Insert filter mode (90deg = 1, 45deg = 3): ";
	cin >> filterMode;

	if (filterMode != 1 && filterMode != 3) {
		cout << "choose a right filter mode!\n";

		exit(-1);
	}

	Mat laplacianImg = laplaciano(img, CV_32F, filterMode);

	imshow("original image", img);
	imshow("laplacian image", laplacianImg);

	waitKey(0);

	return 0;
}