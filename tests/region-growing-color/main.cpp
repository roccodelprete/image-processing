#include <iostream>
#include <opencv2/opencv.hpp>
#include <stack>

using namespace std;
using namespace cv;

Mat grow(Mat src, Mat mask, Mat out, int th, Point seed) {
    Point pointsShift[8] = {
      Point(-1, -1), Point(-1, 0), Point(-1, 1),
      Point(0, -1), Point(0, 1),
      Point(1, -1), Point(1, 0), Point(1, 1)
    };
    
    stack<Point>pointsStack;
    pointsStack.push(seed);
    
    while (!pointsStack.empty()) {
        Point center = pointsStack.top();
        mask.at<uchar>(center) = 1;
        pointsStack.pop();
        
        for (int i = 0; i < 8; i++) {
            Point estimatedPoint = center + pointsShift[i];
            
            if (estimatedPoint.x >= 0 && estimatedPoint.y >= 0 && estimatedPoint.x < src.cols && estimatedPoint.y < src.rows) {
                auto srcCenterPoint = src.at<Vec3b>(center), srcEstimatedPoint = src.at<Vec3b>(estimatedPoint);
                int delta = pow(srcCenterPoint[0] - srcEstimatedPoint[0], 2) + pow(srcCenterPoint[1] - srcEstimatedPoint[1], 2) + pow(srcCenterPoint[2] - srcEstimatedPoint[2], 2);
                
                if (delta < th && out.at<uchar>(estimatedPoint) == 0 && mask.at<uchar>(estimatedPoint) == 0) {
                    mask.at<uchar>(estimatedPoint) = 1;
                    pointsStack.push(estimatedPoint);
                }
            }
        }
    }
    
    return mask;
}

int regionGrowingColor(Mat src, int th, double minRegionFactor, int maxRegionNumber) {
    uchar labels = 1;
    int minRegionArea = int(minRegionFactor * src.rows * src.cols);
    Mat out = Mat::zeros(src.rows, src.cols, CV_8U), mask = Mat::zeros(src.rows, src.cols, CV_8U);
    
    for (int i = 0; i < src.cols; i++) {
        for (int j = 0; j < src.rows; j++) {
            if (out.at<uchar>(i, j) == 0) {
                mask = grow(src, mask, out, th, Point(i, j));
                int maskArea = countNonZero(mask);
                
                if (maskArea > minRegionArea) {
                    out.setTo(labels, mask);
                    labels++;
                    
                    imshow("region", mask * 255);
                    waitKey(0);
                    
                    if (labels > maxRegionNumber) {
                        cout << "oversegmentation" << endl;
                        return -2;
                    }
                }
                mask.setTo(0);
            }
        }
    }
    
    return 0;
}

int main() {
    Mat img = imread("../images/lenna.jpg", IMREAD_COLOR);
    
    if (img.empty()) {
        cout << "Error reading image" << endl;
        return -1;
    }
    
    if (img.rows > 500 || img.cols > 500) {
        resize(img, img, Size(0, 0), .5, .5);
    }
    
    return regionGrowingColor(src, 210, .01, 10);
}
