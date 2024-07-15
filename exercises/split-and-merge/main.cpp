#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define UL_INDEX 0
#define UR_INDEX 1
#define LL_INDEX 2
#define LR_INDEX 3


class TNode {
	Rect region;
	TNode* children[4] = { nullptr };
	vector<TNode*> merged;
	bool is_child_merged[4] = { 0 };
	double stdDeviation, mean;


public:
	TNode(Rect r) { region = r; }


	Rect getRegion() { return region; }
	TNode* getChild(int index) { return children[index]; };
	bool getIsMerged(int index) { return is_child_merged[index]; }
	vector<TNode*> getMerged() { return merged; }
	double getStdDev() { return stdDeviation; }
	double getMean() { return mean; }


	void setChild(int index, TNode* node) { children[index] = node; };
	void setChildMerged(int index) { is_child_merged[index] = true; }
	void addChildMerged(TNode* root) { merged.push_back(root); }
	void setStdDev(double stdDev) { stdDeviation = stdDev; }
	void setMean(double mean) { this->mean = mean; }
};

TNode* split(Mat src, Rect region, double th, int minWidth) {
	TNode* root = new TNode(region);

	Scalar stdDev, mean;

	meanStdDev(src(region), mean, stdDev);
	root->setStdDev(stdDev[0]);
	root->setMean(mean[0]);

	if (region.width > minWidth && root->getStdDev() > th) {
		int halfWidth = region.width / 2;
		int halfHeight = region.height / 2;


		Rect ul = Rect(region.x, region.y, halfHeight, halfWidth);
		Rect ur = Rect(region.x, region.y + halfWidth, halfHeight, halfWidth);
		Rect ll = Rect(region.x + halfHeight, region.y, halfHeight, halfWidth);
		Rect lr = Rect(region.x + halfHeight, region.y + halfWidth, halfHeight, halfWidth);


		root->setChild(UL_INDEX, split(src, ul, th, minWidth));
		root->setChild(UR_INDEX, split(src, ur, th, minWidth));
		root->setChild(LL_INDEX, split(src, ll, th, minWidth));
		root->setChild(LR_INDEX, split(src, lr, th, minWidth));
	}

	rectangle(src, region, 0);

	return root;
}

void doMerge(TNode* root, double th, int minWidth);
bool mergeRegion(TNode* root, vector<int> i, double std_dev_th, int min_width);

void doMerge(TNode* root, double th, int minWidth) {
	if (root->getRegion().width > minWidth && root->getStdDev() > th) {
		if (!(mergeRegion(root, { UL_INDEX, UR_INDEX, LL_INDEX, LR_INDEX }, th, minWidth) ||
			mergeRegion(root, { UR_INDEX, LR_INDEX, UL_INDEX, LL_INDEX }, th, minWidth) ||
			mergeRegion(root, { LL_INDEX, LR_INDEX, UL_INDEX, UR_INDEX }, th, minWidth) ||
			mergeRegion(root, { UL_INDEX, LL_INDEX, UR_INDEX, LR_INDEX }, th, minWidth))) {

			doMerge(root->getChild(UL_INDEX), th, minWidth);
			doMerge(root->getChild(UR_INDEX), th, minWidth);
			doMerge(root->getChild(LL_INDEX), th, minWidth);
			doMerge(root->getChild(LR_INDEX), th, minWidth);
		}
	}
	else {
		root->addChildMerged(root);
		root->setChildMerged(UL_INDEX);
		root->setChildMerged(UR_INDEX);
		root->setChildMerged(LL_INDEX);
		root->setChildMerged(LR_INDEX);
	}
}

bool mergeRegion(TNode* root, vector<int> i, double th, int minWidth) {
	TNode* c0 = root->getChild(i[0]);
	TNode* c1 = root->getChild(i[1]);
	TNode* c2 = root->getChild(i[2]);
	TNode* c3 = root->getChild(i[3]);
	bool merged = false;

	if (c0->getStdDev() <= th && c1->getStdDev() <= th) {
		merged = true;

		root->addChildMerged(c0);
		root->setChildMerged(i[0]);

		root->addChildMerged(c1);
		root->setChildMerged(i[1]);


		if (c2->getStdDev() <= th && c3->getStdDev() <= th) {
			root->addChildMerged(c2);
			root->setChildMerged(i[2]);

			root->addChildMerged(c3);
			root->setChildMerged(i[3]);
			return merged;
		}

		doMerge(c2, th, minWidth);
		doMerge(c3, th, minWidth);
	}

	return merged;
}


void segment(Mat segmented, TNode* root) {
	vector<TNode*> mergedRegions = root->getMerged();

	if (mergedRegions.empty()) {
		for (int i = 0; i < 4; i++) {
			segment(segmented, root->getChild(i));
		}
	}
	else {
		double avgIntensity = 0;

		for (auto region : mergedRegions) {
			avgIntensity += region->getMean();
		}
		avgIntensity /= mergedRegions.size();

		for (auto region : mergedRegions) {
			segmented(region->getRegion()) = (int)avgIntensity;
		}
	}
	if (mergedRegions.size() > 1) {
		for (int i = 0; i < 4; ++i) {
			if (!root->getIsMerged(i)) {
				segment(segmented, root->getChild(i));
			}
		}
	}
}


Mat splitAndMerge(Mat src, double th, int minWidth) {
	int dim = exp2(floor(log2(min(src.cols, src.rows))));

	Rect square = Rect(0, 0, dim, dim);
	Mat cropped = src(square).clone();

	GaussianBlur(cropped, cropped, Size(5, 5), 0);

	TNode* root = split(cropped, Rect(0, 0, cropped.rows, cropped.cols), th, minWidth);

	imshow("splitted and cropped image", cropped);

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

	Mat res = splitAndMerge(img, 8, 6);

	imshow("original", img);
	imshow("split and merge image", res);
	resizeWindow("Immagine segmentata con Split and Merge", res.rows, res.cols);
	waitKey();

	return 0;
}
