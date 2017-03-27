#include "stdafx.h"
#include "class.h"

//测试图像匹配
void ClassTest(){
	
	ReadImage test("D:/2016.10.14衢州水亭门/塔test");
	ReadImage dataSet("D:/2016.10.14衢州水亭门/塔data");

	vector<SiftFeature> DataSet;
	for (int j = 0; j < dataSet.Images.size(); ++j){
		Mat dstMask(dataSet.Images[j].rows, dataSet.Images[j].cols, CV_8UC1, Scalar(255));
		ExtractFeature dstFeature(dataSet.Images[j], dstMask);
		SiftFeature temp = dstFeature.feature;
		DataSet.push_back(temp);
	}


	for (int i = 0; i < test.Images.size(); ++i){
		Mat srcMask(test.Images[i].rows, test.Images[i].cols, CV_8UC1, Scalar(255));
		ExtractFeature srcFeature(test.Images[i], srcMask);

		int maxMatch = 0, index = 0;
		vector<DMatch> bestMatches;
		SiftFeature bestFeatures;
		for (int j = 0; j < dataSet.Images.size(); ++j){
			MatchFeature match(srcFeature.feature, DataSet[j]);
			//match.DrawOutput(test.Images[i], dataSet.Images[j], srcFeature, dstFeature, " ");

			if (match.matches.size()>maxMatch){
				maxMatch = match.matches.size();
				index = j;
				bestMatches = match.matches;
				bestFeatures = DataSet[j];
			}
		}
		namedWindow("BestMatch", WINDOW_AUTOSIZE);
		Mat img_matches;
		//红色连接的是匹配的特征点对，绿色是未匹配的特征点  
		drawMatches(test.Images[i], srcFeature.feature.FeaturePoint, dataSet.Images[index], bestFeatures.FeaturePoint, bestMatches, img_matches, Scalar::all(-1), CV_RGB(0, 255, 0), Mat(), 2);
		imshow("BestMatch", img_matches);
		waitKey(1000);
		destroyWindow("BestMatch");
		
	}
	

}


float ResizeRate(int width, int height, int Max){
	float NewWidth = width > height ? width : height;
	int rate = 1;
	while (NewWidth>Max){
		NewWidth /= 2;
		rate *= 2;
	}
	return 1.0 / rate;
}




Mat ResizeImage(Mat & InputImage,int Max){
	int height = InputImage.rows;
	int width = InputImage.cols;
	float rate = ResizeRate(width,height,Max);
	cout << "W,H,R: " << width << "," << height << "," << rate << endl;
	int NewWidth = width*rate;
	int NewHeight = height*rate;
	int OriginalW = 0;
	int OriginalH = 0;
	Mat OutputImage(NewHeight, NewWidth, CV_8UC3);
	for (size_t i = 0; i < NewHeight;++i){
		for (size_t j = 0; j < NewWidth;++j){
			OriginalH = i / rate;
			OriginalW = j / rate;
			OutputImage.at<Vec3b>(i, j)[0] = InputImage.at<Vec3b>(OriginalH, OriginalW)[0];
			OutputImage.at<Vec3b>(i, j)[1] = InputImage.at<Vec3b>(OriginalH, OriginalW)[1];
			OutputImage.at<Vec3b>(i, j)[2] = InputImage.at<Vec3b>(OriginalH, OriginalW)[2];
		}
	}
	return OutputImage;
}

void getFiles(string path, vector<string>& files) {
	//文件句柄  
	long long hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;   //大家可以去查看一下_finddata结构组成                            
	//以及_findfirst和_findnext的用法，了解后妈妈就再也不用担心我以后不会编了  
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			//如果是目录,迭代之  
			//如果不是,加入列表  
			if ((fileinfo.attrib & _A_SUBDIR)) {
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else {
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0); 
		_findclose(hFile);
	}
}


Mat ExtractFeature::GetMask(int width,int height,int Xleft,int Xright,int Yup,int Ydown){
	Mat mask(height, width, CV_8UC1, Scalar(0));
	for (int i = Xleft; i < Xright;++i){
		for (int j = Yup; j < Ydown; ++j){
			mask.at<unsigned char>(j,i) = 255;
		}
	}
	return mask;
}

ExtractFeature::ExtractFeature(Mat Image,Mat & Mask){
	clock_t start, finish;
	double totaltime;

	//SIFT( int nfeatures=0, int nOctaveLayers=3,double contrastThreshold = 0.04, double edgeThreshold = 10,double sigma = 1.6)
	//nfeatures：特征点数目（算法对检测出的特征点排名，返回最好的features个特征点）。
	//nOctaveLayers：金字塔中每组的层数（算法中会自己计算这个值，后面会介绍）。
	//contrastThreshold：过滤掉较差的特征点的对阈值。contrastThreshold越大，返回的特征点越少。
	//edgeThreshold：过滤掉边缘效应的阈值。edgeThreshold越大，特征点越多（被多滤掉的越少）。
	//sigma：金字塔第0层图像高斯滤波系数，也就是σ。
	SiftFeatureDetector siftdtc(0,3,0.06,6,1.6);
	SiftDescriptorExtractor extractor;

	//Mask = GetMask(Image.cols, Image.rows, Image.cols / 4.0, Image.cols*3.0 / 4.0, Image.rows / 4.0, Image.rows*3.0 / 4.0);

	start = clock();
	siftdtc.detect(Image, feature.FeaturePoint,Mask);
	extractor.compute(Image, feature.FeaturePoint, feature.descriptor);
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "\nfeatures extract takes " << totaltime << " second！" << endl;
}



#if 1
//knn比率测试
//RANSAC方法计算基础矩阵，并细化匹配结果
//计算单应矩阵H，并细化匹配结果
MatchFeature::MatchFeature(SiftFeature f1, SiftFeature f2){
	BruteForceMatcher<L2<float>> matcher;//有问题
	//BFMatcher matcher(NORM_L2, true);//true,交叉过滤
	//DescriptorMatcher matcher;
	vector<vector<DMatch>> knnMatches;//保存knn匹配
	const float minRatio = 0.75f;// 1.f / 1.5f;
	const int k = 2;



	clock_t start, finish;
	double totaltime;
	start = clock();



	matcher.knnMatch(f1.descriptor, f2.descriptor, knnMatches,k);
	for (size_t i = 0; i < knnMatches.size(); i++) {
		const DMatch& bestMatch = knnMatches[i][0];
		const DMatch& betterMatch = knnMatches[i][1];

		float  distanceRatio = bestMatch.distance / betterMatch.distance;
		if (distanceRatio < minRatio)
			matches.push_back(bestMatch);
	}



	//RANSAC方法计算基础矩阵，并细化匹配结果
	/**/
	//Align all points
	vector<KeyPoint> alignedKps1, alignedKps2;
	for (size_t i = 0; i < matches.size(); i++) {
		alignedKps1.push_back(f1.FeaturePoint[matches[i].queryIdx]);
		alignedKps2.push_back(f2.FeaturePoint[matches[i].trainIdx]);
	}

	//Keypoints to points
	vector<Point2f> ps1, ps2;
	for (unsigned i = 0; i < alignedKps1.size(); i++)
		ps1.push_back(alignedKps1[i].pt);

	for (unsigned i = 0; i < alignedKps2.size(); i++)
		ps2.push_back(alignedKps2[i].pt);


	//使用RANSAC方法计算基础矩阵后可以得到一个status向量
	Mat status;
	Mat FundamentalMat = findFundamentalMat(ps1, ps2, FM_RANSAC, 3., 0.99, status);


	//优化匹配结果
	vector<KeyPoint> leftInlier;
	vector<KeyPoint> rightInlier;
	vector<DMatch> inlierMatch;

	int index = 0;
	for (unsigned i = 0; i < matches.size(); i++) {
		if (status.data[i] != 0){
			leftInlier.push_back(alignedKps1[i]);
			rightInlier.push_back(alignedKps2[i]);
			matches[i].trainIdx = index;
			matches[i].queryIdx = index;
			inlierMatch.push_back(matches[i]);
			index++;
		}
	}
	f1.FeaturePoint = leftInlier;
	f2.FeaturePoint = rightInlier;
	matches = inlierMatch;



	//计算单应矩阵H，并细化匹配结果
	const int minNumbermatchesAllowed = 8;
	if (matches.size() < minNumbermatchesAllowed){
		return;
	}
		

	//Prepare data for findHomography
	vector<Point2f> srcPoints(matches.size());
	vector<Point2f> dstPoints(matches.size());
	for (size_t i = 0; i < matches.size(); i++) {
		srcPoints[i] = f1.FeaturePoint[matches[i].trainIdx].pt;
		dstPoints[i] = f2.FeaturePoint[matches[i].queryIdx].pt;
	}

	//find homography matrix and get inliers mask
	vector<uchar> inliersMask(srcPoints.size());
	double reprojectionThreshold = 3.;
	Mat homography = findHomography(srcPoints, dstPoints, CV_FM_RANSAC, reprojectionThreshold, inliersMask);

	vector<DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++){
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}
	matches.swap(inliers);
	



	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "\nmatches' amount:" << matches.size() << endl;
	//cout << "\nfeatures match takes " << totaltime << " second！" << endl;
}

#else

MatchFeature::MatchFeature(ExtractFeature f1, ExtractFeature f2){
	//BruteForceMatcher<L2<float>> matcher;//有问题
	BFMatcher matcher(NORM_L2, true);//true,交叉过滤
	clock_t start, finish;
	double totaltime;
	start = clock();
	matcher.match(f1.descriptor, f2.descriptor,matches);


	//反向匹配，看是否匹配自身
	/*vector<DMatch> backMatches;
	matcher.match( f2.descriptor,f1.descriptor, backMatches);
	//筛选出较好的匹配点  
	vector<DMatch> goodMatches;
	for (int i = 0; i<matches.size(); i++){
		if (backMatches[matches[i].trainIdx].trainIdx == i){
			goodMatches.push_back(matches[i]);
		}
	}
	cout << "原来Match个数：" << matches.size() << endl;
	matches.swap(goodMatches);
	cout << "好的Match个数：" << matches.size() << endl;
	goodMatches.clear();*/



	//计算匹配结果中距离的最大和最小值  
	//距离是指两个特征向量间的欧式距离，表明两个特征的差异，值越小表明两个特征点越接近  
	/*double max_dist = 0;
	double min_dist = 100;
	for (int i = 0; i<matches.size(); i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	cout << "最大距离：" << max_dist << endl;
	cout << "最小距离：" << min_dist << endl;
	double proportion = 0.31;
	double threshold = min_dist + proportion*(max_dist - min_dist);
	//筛选出较好的匹配点  
	//vector<DMatch> goodMatches;
	for (int i = 0; i<matches.size(); i++){
		if (matches[i].distance <= threshold){
			goodMatches.push_back(matches[i]);
		}
	}
	cout << "原来Match个数：" << matches.size() << endl;
	matches.swap(goodMatches);
	cout << "好的Match个数：" << matches.size() << endl;
	*/


	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "\nmatches' amount:" << matches.size() << endl;
	//cout << "\nfeatures match takes " << totaltime << " second！" << endl;
}

#endif


void MatchFeature::DrawOutput(Mat Image1, Mat Image2, SiftFeature f1, SiftFeature f2, const string path){
	namedWindow("Matches", WINDOW_AUTOSIZE);
	Mat img_matches;
	//drawMatches(Image1, f1.FeaturePoint,Image2, f2.FeaturePoint, matches, img_matches);

	//红色连接的是匹配的特征点对，绿色是未匹配的特征点  
	drawMatches(Image1, f1.FeaturePoint, Image2, f2.FeaturePoint, matches, img_matches, Scalar::all(-1)/*CV_RGB(255,0,0)*/, CV_RGB(0, 255, 0), Mat(), 2);



	imshow("Matches", img_matches);
	waitKey(0);
	destroyWindow("Matches");
}


ReadImage::ReadImage(const string path){//可以将图片降采样来提高速度
	//cvNamedWindow("video", 1);
	vector<string> files;
	getFiles(path, files);
	for (size_t i = 0; i < files.size(); i++) {    //files.size()返回文件数量  
		Mat Image = imread(files[i]);
		Mat ResizedImage;
		if (  (Image.size().width > Image.size().height ? Image.size().width : Image.size().height)  > MaxWidth  ){
			//float rate = ResizeRate(Image.size().width, Image.size().height);
			//resize(Image, ResizedImage, Size(0, 0), rate, rate, INTER_NEAREST);
			if (Image.size().width>Image.size().height)
				resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / Image.size().width, 1.0*MaxWidth / Image.size().width, INTER_LINEAR);
			else
				resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / Image.size().height, 1.0*MaxWidth / Image.size().height, INTER_LINEAR);
			
			Images.push_back(ResizedImage);
			//imshow("video", ResizedImage);
		}
		else{
			Images.push_back(Image);
			//imshow("video",Image);
		}
		//waitKey(100);
		cout << files[i] << endl;
	}
	cout <<Images.size() << " images are read!\n" << endl;
	//destroyWindow("video");
}