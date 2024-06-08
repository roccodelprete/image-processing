#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct Offset {
    int x;
    int y;
};

unsigned char calcFilteredValue(Mat paddedImage, Mat mask, Offset offset) {
    float value = 0;

    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            value += paddedImage.at<unsigned char>(i + offset.x, j + offset.y) * mask.at<float>(i, j);
        }
    }

    return (unsigned char)value;
}

Mat correlation(Mat src, Mat mask) {
    Mat out(src.rows, src.cols, src.type());
    Mat paddedImage;
    int borderWidth = mask.rows / 2;
    copyMakeBorder(src, paddedImage, borderWidth, borderWidth, borderWidth, borderWidth, BORDER_REFLECT);

    Offset offset;

    for (int i = 0; i < out.rows; i++) {
        for (int j = 0; j < out.cols; j++) {
            offset.x = i;
            offset.y = j;

            out.at<unsigned char>(i, j) = calcFilteredValue(paddedImage, mask, offset);
        }
    }

    return out;
}

Mat convulation(Mat src, Mat mask) {
    Mat temp;
    rotate(mask, temp, ROTATE_180);

    return correlation(src, temp);
}

int main() {
    Mat image = imread("../../images/cameraman.tif", IMREAD_GRAYSCALE);
    if (image.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    int dim;
    cout << "Insert dimensions (it must be odd): ";
    cin >> dim;

    if (dim % 2 == 0) {
        cout << "Dimension must be odd!" << endl;
        return -1;
    }

    Mat mask = Mat::ones(dim, dim, CV_32F) / (float)(dim * dim);

    Mat filter2DImage;
    filter2D(image, filter2DImage, image.type(), mask);

    Mat correlationImage = correlation(image, mask);
    Mat convulationImage = convulation(image, mask);

    imshow("Original", image);
    imshow("Filter 2D", filter2DImage);
    imshow("Correlation", correlationImage);
    imshow("Convolution", convulationImage);

    waitKey(0);

    return 0;
}
