#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

/**
* This function applies the Laplacian of Gaussian to the source image
* The Laplacian of Gaussian marks image borders
* @param src Source image
* @param depth Depth of the output image
* @param ksize Size of filter
* @returns The image with LoG applied
*/
Mat LoG(Mat src, int depth, int ksize) {
    Mat output, gaussianImg;

    GaussianBlur(src, gaussianImg, Size(ksize, ksize), 0, 0);
    Laplacian(gaussianImg, output, depth, ksize);
    
    return output;
}

/**
* This function add the LoG mask to source image
* @param src Source image
* @param value Scaling value for the Laplacian of Gaussian mask.
* It must be -1 <= value <= 0.
* @returns The image with LoG applied summed to a constat
*/
Mat sharpening(Mat src, float value) {
    Mat LoGImg = LoG(src, CV_8U, 1);

    return (src + value * LoGImg);
}

int main() {

    Mat img = imread("../images/building.png", IMREAD_GRAYSCALE);
    Mat LoGImg = LoG(img, CV_8U, 3);
    Mat sharpedLoGImg = sharpening(img, -1);

    imshow("original", img);
    imshow("LoG", LoGImg);
    imshow("sharped LoG", sharpedLoGImg);

    waitKey(0);

    return 0;
}