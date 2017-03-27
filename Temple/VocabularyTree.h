#ifndef VocabularyTree_H
#define VocabularyTree_H

#include "head.h"
#include "class.h"

#define K 5 //k-means种类数
#define MaxIter 10 //k-means最大迭代次数
#define Bias 1.0 //k-means中心的偏移比例结束条件
#define MinCluster 30//定义叶子节点最多包含的描述子数目

#define OrbFlag 1//1:orb; 0:其他
#define FeatureFlag 0//1：sift; 0:surf;
#if OrbFlag

#define D 32 //orb描述子的维度
#define ORBnfeatures 1000//orb特征点的最多个数
#else

#if FeatureFlag
#define D 128 //sift描述子的维度
#else 
#define D 64 //surf描述子的维度
#endif

#endif




struct VtreeData{
	float CenterDesc[D];
	vector<int> DescIndex;
};
struct Vtree{
	Vtree *next[K];
	VtreeData data;
};

struct ImageData{
	Mat Image;
	vector<KeyPoint> FeaturePoint;
	string file;
};
struct Desc{
	float descriptor[D];
};

class VT{
public:
	
	VT(const string & path);

	~VT(){}
	
	Mat mergeRows(Mat A, Mat B);//矩阵按行合并

	float DescDist(float a[], float b[]);

	float DescDist(int index, float b[]);

	float DescDist(float b[], int index);

	void CreateVtree(Vtree * root);

	void SearchDesc(float a[], vector<float> &marker);

	void Show2Image(Mat & Img1, Mat & Img2);

	int MatchFeatureNumber(const vector<KeyPoint> &kp1, const vector<KeyPoint> &kp2, const Mat &d1, const Mat &d2, vector<DMatch> &matches);

	void SearchImage(string file);


private:
	string ImageSetPath;
	Vtree * root;//词汇树
	vector<ImageData> images;
	Mat AllDesc;
	vector<int> ImageOfDesc;
};


#endif