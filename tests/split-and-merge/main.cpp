#include <iostream>
#include <opencv2/opencv.hpp>

#define UL 0
#define UR 1
#define LL 2
#define LR 3

using namespace std;
using namespace cv;

void doMerge(TNode*, double, int);
bool mergeRegion(TNode*, vector<int>, double, int);

class TNode {
    Rect region;
    double stdDev, mean;
    vector<TNode*> merged;
    bool isMerged[4] = { false };
    TNode* children[4] = { nullptr };
    
    TNode(Rect r) : region(r) {};
    Rect getRegion() { return this->region; }
    double getStdDeev() { return this->stdDev; }
    double getMean() { return this->mean; }
    bool getIsMerged(int index) { return this->isMerged[index]; }
    TNode* getChild(int index) { return this->children[index]; }
    vector<TNode*> getMerged() { return this->merged; }
    
    void setChild(int index, TNode* node) { this->children[index] = node; }
    void setIsMerged(int index) { this->isMerged[index] = true; }
    void setMean(double mean) { this->mean = mean; }
    void setStdDev(double stdDev) { this->stdDev = stdDev; }
    void addChildMerged(TNode* root) { this->merged.push_back(root); }
};

TNode* split(Mat src, Rect region, double th, int minWidth) {
    TNode* root = new TNode(region);
    Scalar mean, stdDev;
    meanStdDev(mean, stdDev);
    root->setMean(mean[0]);
    root->setStdDev(stdDev[0]);
    
    if (region.width > minWidth && root->getStdDev() > th) {
        int halfWidth = region.width / 2;
        int halfheight = region.height / 2;
        
        root->setChild(UL, split(src, Rect(region.x, region.y, halfHeight, halfWidth), th, minWidth));
        root->setChild(UR, split(src, Rect(region.x, region.y + halfWidth, halfHeight, halfWidth), th, minWidth));
        root->setChild(LL, split(src, Rect(region.x + halfHeight, region.y, halfHeight, halfWidth), th, minWidth));
        root->setChild(UR, split(src, Rect(region.x + halfHeight, region.y + halfWidth, halfHeight, halfWidth), th, minWidth));
    }
    
    rectangle(src, region, 0);
    
    return root;
}

bool mergeRegion(TNode* root, vector<int> indices, double th, int minWidth) {
    TNode* c0 = root->getChild(indices[0]);
    TNode* c1 = root->getChild(indices[1]);
    TNode* c2 = root->getChild(indices[2]);
    TNode* c3 = root->getChild(indices[3]);
    
    if (c0->getStdDev() <= th && c1->getStdDev() <= th) {
        root->addChildMerged(c0);
        root->setIsMerged(indices[0]);
        
        root->addChildMerged(c1);
        root->setIsMerged(indices[1]);
        
        if (c2->getStdDev() <= th && c3->getStdDev() <= th) {
            root->addChildMerged(c2);
            root->setIsMerged(indices[2]);
            
            root->addChildMerged(c3);
            root->setIsMerged(indices[3]);
        }
        else {
            doMerge(c2, th, minWidth);
            doMerge(c3, th, minWidth);
        }
    }
}

void doMerge(TNode* root, double th, int minWidth) {
    if (root->getRegion().width > minWidth && root->getStdDev() > th) {
        if (!(mergeRegion(root, { UL, UR, LL, LR }, th, minWidth) ||
              mergeRegion(root, { UR, LR, UL, LL }, th, minWidth) ||
              mergeRegion(root, { LL, LR, UL, UR }, th, minWidth) ||
              mergeRegion(root, { UL, LL, UR, LR}, th, minWidth)
              )) {
            for (int i 0 0; i < 4; i++) {
                doMerge(root-getChild(i), th, minWidth);
            }
        }
    }
    else {
        root->addChildMerged(root);
        
        for (int i = 0; i < 4; i++) {
            root->setIsMerged(i);
        }
    }
}

void segment(Mat &segmented, TNode* root) {
    vector<TNode*> mergedRegions = root->getMerged();
    
    if (mergedRegion.empty()) {
        for (int i = 0; i < 4; i++) {
            segment(segmented, root);
        }
    }
    else {
        double avgIntensity = 0.0;
        
        for (auto region : mergedRegions) {
            avgIntensity += region->getMean();
        }
        avgIntensity /= mergedRegions.size();
        
        for (auto region : mergedRegions) {
            segmented(region->getRegion()) = int(avgIntensity);
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
    int dim = exp2(floor(log(min(src.cols, src.rows))));
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
