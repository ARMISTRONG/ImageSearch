#ifndef VocabularyTree_H
#define VocabularyTree_H

#include "head.h"
#include "class.h"

#define K 5 //k-means������
#define MaxIter 10 //k-means����������
#define Bias 1.0 //k-means���ĵ�ƫ�Ʊ�����������
#define MinCluster 30//����Ҷ�ӽڵ�����������������Ŀ

#define OrbFlag 1//1:orb; 0:����
#define FeatureFlag 0//1��sift; 0:surf;
#if OrbFlag

#define D 32 //orb�����ӵ�ά��
#define ORBnfeatures 1000//orb�������������
#else

#if FeatureFlag
#define D 128 //sift�����ӵ�ά��
#else 
#define D 64 //surf�����ӵ�ά��
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
	
	Mat mergeRows(Mat A, Mat B);//�����кϲ�

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
	Vtree * root;//�ʻ���
	vector<ImageData> images;
	Mat AllDesc;
	vector<int> ImageOfDesc;
};


#endif