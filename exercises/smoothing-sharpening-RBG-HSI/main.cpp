#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
* Implementare smoothing e sharpening sulle componenti R, G e B
* Confrontare il risultato con quello ottenuto modificando solo la componente
* di intensità della rappresentazione HSI
*/

Mat smoothingRGB(Mat src, int size) {
	// Mat bgr[3];				// Matrix with 3 channels R G B

	// split(src, bgr);			// split divides matrix in three channels: B (blue), G (green), R (red)

	/*
		blur(bgr[0], bgr[0], Size(size, size));
		blur(bgr[1], bgr[1], Size(size, size));
		blur(bgr[2], bgr[2], Size(size, size));

		Mat out;
		merge(bgr, 3, out);
	*/
	Mat out;

	blur(src, out, Size(size, size));

	return out;
}

Mat sharpeningRGB(Mat src) {
	// Mat bgr[3];			// Matrix with 3 channels R G B

	// split(src, bgr);		// split divides matrix in three channels: B (blue), G (green), R (red)

	/*
		Laplacian(bgr[0], bgr[0], bgr[0].type(), 3);
		Laplacian(bgr[1], bgr[1], bgr[1].type(), 3);
		Laplacian(bgr[2], bgr[2], bgr[2].type(), 3);
		
		Mat out;
		merge(bgr, 3, out);
	*/

	Mat out;
	Laplacian(src, out, src.type(), 3);

	return out;
}

Mat smoothingHSV(Mat src, int size) {
	Mat bgr[3];

	split(src, bgr);

	blur(bgr[2], bgr[2], Size(size, size));

	Mat out;
	merge(bgr, 3, out);

	cvtColor(out, out, COLOR_HSV2BGR);

	return out;
}

Mat sharpeningHSV(Mat src) {
	Mat bgr[3];

	split(src, bgr);

	Laplacian(bgr[2], bgr[2], bgr[2].type(), 3);

	Mat out;
	merge(bgr, 3, out);

	cvtColor(out, out, COLOR_HSV2BGR);

	return out;
}

Mat rgbToHsv(Mat src) {
	Mat out;
	cvtColor(src, out, COLOR_BGR2HSV);

	return out;
}

int main() {
	int size;
	cout << "Insert size: ";
	cin >> size;
	
	Mat img = imread("../../images/lenna.jpg", IMREAD_COLOR);
	Mat imgHSV = rgbToHsv(img);
	
	Mat smoothedImgRGB = smoothingRGB(img, size);
	Mat sharpedImgRGB = sharpeningRGB(img);
	Mat smoothedImgHSV = smoothingHSV(imgHSV, size);
	Mat sharpedImgHSV = sharpeningHSV(imgHSV);
	
	imshow("original", img);
	imshow("original HSV", imgHSV);
	imshow("smoothing RGB", smoothedImgRGB);
	imshow("sharpening RGB", sharpedImgRGB);
	imshow("smoothing HSV", smoothedImgHSV);
	imshow("sharpening HSV", sharpedImgHSV);

	waitKey(0);

	return 0;
}