#ifndef CLASS_H
#define CLASS_H


#include "head.h"


#define  MaxWidth 640


float ResizeRate(int width, int height, int Max = MaxWidth);

Mat ResizeImage(Mat & InputImage, int Max = MaxWidth);

void getFiles(string path, vector<string>& files);


struct SiftFeature{
	vector<KeyPoint> FeaturePoint;
	Mat descriptor;
};


class ExtractFeature{
public:
	friend class MatchFeature;

	SiftFeature feature;

	ExtractFeature(Mat Image, Mat & Mask);

	~ExtractFeature(){}

private:
	Mat GetMask(int width, int height, int Xleft, int Xright, int Yup, int Ydown);

};


	class MatchFeature{
	public:
		vector<DMatch> matches;

		MatchFeature(SiftFeature f1, SiftFeature f2);

		~MatchFeature(){}

		void DrawOutput(Mat Image1, Mat Image2, SiftFeature f1, SiftFeature f2, const string path);

	private:
		
	};


	class ReadImage{
	public:
		vector<Mat> Images;

		ReadImage(const string path);

		~ReadImage(){}

	private:
		

	};

	void ClassTest();

#endif