#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define UL 0
#define UR 1
#define LL 2
#define LR 3

class TNode {
    Rect region;
    TNode* children[4] = { nullptr };
    vector<TNode*> merged;
    bool isMerged[4] = { false };
    double stdDev, mean;

public:
    TNode(Rect r) : region(r) {}
    Rect getRegion() { return region; }
    TNode* getChild(int index) { return children[index]; }
    bool getIsMerged(int index) { return isMerged[index]; }
    vector<TNode*> getMerged() { return merged; }
    double getStdDev() { return stdDev; }
    double getMean() { return mean; }

    void setChild(int index, TNode* node) { children[index] = node; }
    void setChildMerged(int index) { isMerged[index] = true; }
    void addChildMerged(TNode* root) { merged.push_back(root); }
    void setStdDev(double stdDev) { this->stdDev = stdDev; }
    void setMean(double mean) { this->mean = mean; }
};

TNode* split(Mat src, Rect region, double th, int minWidth) {
    TNode* root = new TNode(region);
    Scalar mean, stdDev;
    meanStdDev(src(region), mean, stdDev);
    root->setStdDev(stdDev[0]);
    root->setMean(mean[0]);

    if (region.width > minWidth && root->getStdDev() > th) {
        int halfWidth = region.width / 2;
        int halfHeight = region.height / 2;

        root->setChild(UL, split(src, Rect(region.x, region.y, halfHeight, halfWidth), th, minWidth));
        root->setChild(UR, split(src, Rect(region.x, region.y + halfWidth, halfHeight, halfWidth), th, minWidth));
        root->setChild(LL, split(src, Rect(region.x + halfHeight, region.y, halfHeight, halfWidth), th, minWidth));
        root->setChild(LR, split(src, Rect(region.x + halfHeight, region.y + halfWidth, halfHeight, halfWidth), th, minWidth));
    }

    rectangle(src, region, 0);
    return root;
}

bool mergeRegion(TNode* root, vector<int> indices, double th, int minWidth) {
    TNode* c0 = root->getChild(indices[0]);
    TNode* c1 = root->getChild(indices[1]);
    TNode* c2 = root->getChild(indices[2]);
    TNode* c3 = root->getChild(indices[3]);
    bool merged = false;

    if (c0->getStdDev() <= th && c1->getStdDev() <= th) {
        merged = true;
        root->addChildMerged(c0);
        root->setChildMerged(indices[0]);
        root->addChildMerged(c1);
        root->setChildMerged(indices[1]);

        if (c2->getStdDev() <= th && c3->getStdDev() <= th) {
            root->addChildMerged(c2);
            root->setChildMerged(indices[2]);
            root->addChildMerged(c3);
            root->setChildMerged(indices[3]);
        } else {
            doMerge(c2, th, minWidth);
            doMerge(c3, th, minWidth);
        }
    }
    
    return merged;
}

void doMerge(TNode* root, double th, int minWidth) {
    if (root->getRegion().width > minWidth && root->getStdDev() > th) {
        if (!(mergeRegion(root, { UL, UR, LL, LR }, th, minWidth) ||
              mergeRegion(root, { UR, LR, UL, LL }, th, minWidth) ||
              mergeRegion(root, { LL, LR, UL, UR }, th, minWidth) ||
              mergeRegion(root, { UL, LL, UR, LR }, th, minWidth))) {
            for (int i = 0; i < 4; i++) {
                doMerge(root->getChild(i), th, minWidth);
            }
        }
    } else {
        root->addChildMerged(root);
        for (int i = 0; i < 4; i++) {
            root->setChildMerged(i);
        }
    }
}

void segment(Mat& segmented, TNode* root) {
    vector<TNode*> mergedRegions = root->getMerged();
    if (mergedRegions.empty()) {
        for (int i = 0; i < 4; i++) {
            segment(segmented, root->getChild(i));
        }
    } else {
        double avgIntensity = 0.0;
        for (auto region : mergedRegions) {
            avgIntensity += region->getMean();
        }
        avgIntensity /= mergedRegions.size();

        for (auto region : mergedRegions) {
            segmented(region->getRegion()) = (int)avgIntensity;
        }
    }

    if (mergedRegions.size() > 1) {
        for (int i = 0; i < 4; i++) {
            if (!root->getIsMerged(i)) {
                segment(segmented, root->getChild(i));
            }
        }
    }
}

Mat splitAndMerge(Mat src, double th, int minWidth) {
    int dim = exp2(floor(log2(min(src.cols, src.rows))));
    Mat cropped = src(Rect(0, 0, dim, dim)).clone();
    GaussianBlur(cropped, cropped, Size(5, 5), 0);

    TNode* root = split(cropped, Rect(0, 0, cropped.rows, cropped.cols), th, minWidth);
    doMerge(root, th, minWidth);

    Mat segmented = cropped.clone();
    segment(segmented, root);

    return segmented;
}

int main() {
    Mat img = imread("../images/cameraman.tif", IMREAD_GRAYSCALE);
    
    if (img.empty()) {
        cout << "Error reading image" << endl;
        return -1;
    }
    
    imshow("original", img);
    imshow("split and merge image", splitAndMerge(img, 8, 6));
    waitKey();
    
    return 0;
}
