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
	//Mat descriptor;
	string file;
	int FeatureCounter;
};
struct Desc{
	float descriptor[D];
	int ImageIndex;
};

class VT{
public:
	
	VT(const string & path);

	~VT(){}

	void CoutDesc(float a[]);

	float DescDistance(float a[], float b[]);

	void DescMean(vector<VtreeData> &input);

	int Nearest(int InputIndex, vector<VtreeData> cluster);

	vector<VtreeData> K_Means(int k, VtreeData input);

	void CreateVtree(Vtree * root);

	void SearchDesc(float a[], vector<float> &marker);

	void Show2Image(Mat & Img1, Mat & Img2);

	void SearchImage(string file);




private:
	string ImageSetPath;
	Vtree * root;//词汇树
	vector<ImageData> images;
	vector<Desc> descriptors;


};


#endif