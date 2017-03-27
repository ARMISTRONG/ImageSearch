#ifndef ORBFEATURE_H
#define ORBFEATURE_H


#include "head.h"
#include "class.h"
#include "VocabularyTree.h"


class OrbFeature{
public:
	Mat ReadImage(string path){
		Mat Image = imread(path);
		Mat ResizedImage;
		if ((Image.size().width > Image.size().height ? Image.size().width : Image.size().height)  > MaxWidth){
			//float rate = ResizeRate(Image.size().width, Image.size().height);
			//resize(Image, ResizedImage, Size(0, 0), rate, rate, INTER_NEAREST);
			if (Image.size().width>Image.size().height)
				resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / Image.size().width, 1.0*MaxWidth / Image.size().width, INTER_NEAREST);
			else
				resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / Image.size().height, 1.0*MaxWidth / Image.size().height, INTER_NEAREST);
			
			return ResizedImage;
		}
		else{
			return Image;
		}
	}


	OrbFeature(string path){
		getFiles(path,files);
		int ImageCounter = files.size();
		for (int i = 0; i < ImageCounter;++i){
			images.push_back(ReadImage(files[i]));
			/*ORB(int nfeatures = 500,
			float scaleFactor = 1.2f,
			int nlevels = 8,
			int edgeThreshold = 31,
			int firstLevel = 0,
			int WTA_K = 2,
			int scoreType = ORB::HARRIS_SCORE,
			int patchSize = 31);*/
			ORB orb(ORBnfeatures);
			vector<KeyPoint> points;
			Mat desc;
			orb(images[i],Mat(),points,desc);//Mask为空
			descriptors.push_back(desc);
			featurePoints.push_back(points);
			cout << files[i] << endl;
			cout << "type,dimension:" << desc.type() << "," << desc.cols << endl;
		}
	}


	void ShowResult(Mat & Img1, Mat & Img2){
		Mat img1, img2;
		resize(Img1, img1, Size(0, 0), 1.0*MaxWidth / Img1.cols, 1.0*MaxWidth / Img1.cols, INTER_NEAREST);
		resize(Img2, img2, Size(0, 0), 1.0*MaxWidth / Img2.cols, 1.0*MaxWidth / Img2.cols, INTER_NEAREST);

		Mat expanded(Size((img1.cols + img2.cols), img1.rows>img2.rows ? img1.rows : img2.rows), CV_8UC3);
		Mat ROI = expanded(Rect(0, 0, img1.cols, img1.rows));
		Mat ROI1 = expanded(Rect(img1.cols, 0, img2.cols, img2.rows));
		addWeighted(ROI, 0, img1, 1, 0., ROI);
		addWeighted(ROI1, 0, img2, 1, 0., ROI1);

		namedWindow("match result");
		imshow("match result", expanded);
		waitKey(0);
		destroyWindow("match result");
	}


	int BruteSearch(string path){
		Mat testImage = ReadImage(path);
		ORB orb(ORBnfeatures);
		vector<KeyPoint> points;
		Mat desc;
		orb(testImage, Mat(), points, desc);//Mask为空

		int bestMatchIndex = 0;
		float bestMatchRate = 0.0;

		for (size_t i = 0; i < images.size();++i){
			BruteForceMatcher<HammingLUT> matcher;
			vector<DMatch> matches;
			matcher.match(desc, descriptors[i], matches);

			double max_dist = 0; double min_dist = 100;
			//-- Quick calculation of max and min distances between keypoints     
			for (int i = 0; i < desc.rows; i++)
			{
				double dist = matches[i].distance;
				if (dist < min_dist) min_dist = dist;
				if (dist > max_dist) max_dist = dist;
			}
			printf("-- Max dist : %f \n", max_dist);
			printf("-- Min dist : %f \n", min_dist);
			//-- Draw only "good" matches (i.e. whose distance is less than 0.6*max_dist )     
			//-- PS.- radiusMatch can also be used here.     
			std::vector< DMatch > good_matches;
			for (int i = 0; i < desc.rows; i++)
			{
				if (matches[i].distance < 0.6*max_dist)
				{
					good_matches.push_back(matches[i]);
				}
			}
			//Mat img_matches;
			//drawMatches(testImage, points, images[i], featurePoints[i], good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
			//imshow("Match", img_matches);
			//waitKey(0);

			float tempRate = float(good_matches.size()) / featurePoints[i].size();
			if (tempRate>bestMatchRate){
				bestMatchIndex = i;
				bestMatchRate = tempRate;
			}
		}
		ShowResult(testImage, images[bestMatchIndex]);

	}


private:
	vector<string> files;
	vector<Mat> images;
	vector<Mat> descriptors;
	vector<vector<KeyPoint>> featurePoints;
};

void OrbFeatureTest(){
	OrbFeature test("D:/2016.10.14衢州水亭门/塔data");
	vector<string> files;
	getFiles("D:/2016.10.14衢州水亭门/塔test", files);
	for (size_t i = 0; i < files.size(); ++i){
		test.BruteSearch(files[i]);
	}
}

#endif