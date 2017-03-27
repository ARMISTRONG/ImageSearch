#include "stdafx.h"
#include "VocabularyTree.h"

/*
INTER_NEAREST = CV_INTER_NN, //!< nearest neighbor interpolation
INTER_LINEAR = CV_INTER_LINEAR, //!< bilinear interpolation
INTER_CUBIC = CV_INTER_CUBIC, //!< bicubic interpolation
INTER_AREA = CV_INTER_AREA, //!< area-based (or super) interpolation
INTER_LANCZOS4 = CV_INTER_LANCZOS4, //!< Lanczos interpolation over 8x8 neighborhood
INTER_MAX = 7,
WARP_INVERSE_MAP = CV_WARP_INVERSE_MAP
*/

//矩阵按行合并
Mat VT::mergeRows(Mat A, Mat B){
	if (A.cols == B.cols && A.type() == B.type()){
		int totalRows = A.rows + B.rows;
		Mat mergedDescriptors(totalRows, A.cols, A.type());
		Mat submat = mergedDescriptors.rowRange(0, A.rows);
		A.copyTo(submat);
		submat = mergedDescriptors.rowRange(A.rows, totalRows);
		B.copyTo(submat);
		return mergedDescriptors;
	}
	else{
		cout << "cannot merge two Mat!" << endl;
		return A;
	}
}


VT::VT(const string & path){
	ImageSetPath = path;
	vector<string> files;
	getFiles(path, files);
	

	//计时
	clock_t start, finish;
	double totaltime;
	start = clock();

	for (size_t i = 0; i < files.size(); i++) {    //files.size()返回文件数量  
		ImageData temp;
		temp.file = files[i];
		Mat Image = imread(files[i]);
		int width = Image.cols;
		int height = Image.rows;
		Mat ResizedImage;

		if ((width > height ? width : height)  > MaxWidth){
			//float rate = ResizeRate(width, height);
			//resize(Image, ResizedImage, Size(0, 0), rate, rate, INTER_NEAREST);
			if (width>height)
				resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / width, 1.0*MaxWidth / width, INTER_LINEAR);
			else
				resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / height, 1.0*MaxWidth / height, INTER_LINEAR);
			
			temp.Image = ResizedImage;
		}
		else{
			temp.Image = Image;
		}
		temp.Image = Image;


#if OrbFlag
		/*ORB(int nfeatures = 500, 
			float scaleFactor = 1.2f, 
			int nlevels = 8, 
			int edgeThreshold = 31, 
			int firstLevel = 0, 
			int WTA_K = 2, 
			int scoreType = ORB::HARRIS_SCORE, 
			int patchSize = 31);*/
		ORB extractor(ORBnfeatures);
		Mat descriptor;//临时存放一张图片的描述子,descriptor.type():0:CV_8U
		extractor(temp.Image, Mat(),temp.FeaturePoint, descriptor);
		if (i == 0){
			AllDesc = descriptor;
			ImageOfDesc.clear();
		}
		else{
			AllDesc = mergeRows(AllDesc,descriptor);
		}
		for (size_t j = 0; j < descriptor.rows; ++j){
			ImageOfDesc.push_back(i);
		}
		images.push_back(temp);
	}
#endif
		
	cout << images.size() << " images are read!" << endl;
	cout << AllDesc.rows << " descriptors are created!" << endl;


	//计时
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "features extract takes " << totaltime << " seconds!" << endl;




	{
		root = new Vtree;
		for (size_t i = 0; i < AllDesc.rows; ++i){
			root->data.DescIndex.push_back(i);
		}
		clock_t start, finish;
		double totaltime;
		start = clock();
		CreateVtree(root);
		cout << "\nVtree created!" << endl;
		finish = clock();
		totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
		cout << "building the vacabulary tree takes " << totaltime << " seconds！\n" << endl;
	}

}



float VT::DescDist(float a[], float b[]){
	float dis = 0;
	for (int i = 0; i < D; ++i){
		if (a[i] == b[i]) continue;
		dis += pow(a[i] - b[i], 2);
	}
	dis = pow(dis, 0.5);
	return dis;
}


float VT::DescDist(int index, float b[]){
	float dis = 0;
	for (int i = 0; i < D; ++i){
		if ( (float)AllDesc.at<unsigned char>(index,i) == b[i]) continue;
		dis += pow((float)AllDesc.at<unsigned char>(index, i) - b[i], 2);
	}
	dis = pow(dis, 0.5);
	return dis;
}


float VT::DescDist(float b[], int index){
	float dis = 0;
	for (int i = 0; i < D; ++i){
		if ((float)AllDesc.at<unsigned char>(index, i) == b[i]) continue;
		dis += pow((float)AllDesc.at<unsigned char>(index, i) - b[i], 2);
	}
	dis = pow(dis, 0.5);
	return dis;
}



//用opencv自带的一些函数
void VT::CreateVtree(Vtree * root){
	if (root->data.DescIndex.size() <= MinCluster){
		for (int i = 0; i < K; ++i){
			root->next[i] = NULL;
		}
	}
	else{
		//RNG rng(12345);//使用opencv的RNG随机,Multiply-with-Carry algorithm

		//! clusters the input data using k-Means algorithm
		//CV_EXPORTS_W double kmeans(InputArray data, int K, CV_OUT InputOutputArray bestLabels,
		//	TermCriteria criteria, int attempts,
		//	int flags, OutputArray centers = noArray());

		/*samples: (input) The actual data points that you need to cluster. It should contain exactly one point per row. That is, if you have 50 points in a 2D plane, then you should have a matrix with 50 rows and 2 columns.
		clusterCount: (input) The number of clusters in the data points.
		labels: (output) Returns the cluster each point belongs to. It can also be used to indicate the initial guess for each point.
		termcrit: (input) This is an iterative algorithm. So you need to specify the termination criteria (number of iterations & desired accuracy)
		attempts: (input) The number of times the algorithm is run with different center placements
		flags: (input) Possible values include:
		KMEANS_RANDOM_CENTER: Centers are generated randomly
		KMEANS_PP_CENTER: Uses the kmeans++ center initialization
		KMEANS_USE_INITIAL_LABELS: The first iteration uses the suppliedlabels to calculate centers. Later iterations use random or semi-random centers (use the above two flags for that).
		centers: (output) This matrix holds the center of each cluster.*/

		int size = root->data.DescIndex.size();
		Mat points(size,D,CV_32F,Scalar(0.));
		for (int i = 0; i < size;++i){
			for (int j = 0; j < D;++j){
				points.at<float>(i, j) = AllDesc.at<unsigned char>(root->data.DescIndex[i],j);
			}
		}
		Mat lables,centers;
		kmeans(points, K, lables, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);//lables.type():4:CV_32S;centers.type():5:CV_32F

		vector<VtreeData> result(K);
		for (int i=0; i < K;++i){
			result[i].DescIndex.clear();
			for (int j = 0; j < D; ++j){
				result[i].CenterDesc[j] = centers.at<float>(i, j);
			}
		}
		for (int i = 0; i<size; ++i){
			result[lables.at<int>(i, 0)].DescIndex.push_back(root->data.DescIndex[i]);
		}

		//输出分类结果
		/*for (int i = 0; i<K; ++i){
			cout << result[i].DescIndex.size()<<" ";
		}
		cout << endl;*/

		for (int i = 0; i < K; ++i){
			//cout << result[i].DescIndex.size() << endl;
			root->next[i] = new Vtree;
			root->next[i]->data = result[i];
			CreateVtree(root->next[i]);
		}
	}
	return;
}



void VT::SearchDesc(float a[], vector<float> &marker){
	Vtree * test = root;
	float Min=1000000,Max=0,DisThreshold = 1000000;
	while (test->next[0] != NULL){
		float MinDis = 1000000;
		int MinIndex = 0;
		float DisTemp = 0;
		for (int i = 0; i < K; ++i){
			DisTemp = DescDist(a, test->next[i]->data.CenterDesc);
			if (DisTemp<MinDis){
				MinDis = DisTemp;
				MinIndex = i;
			}
		}
		for (size_t i = 0; i < K-1; ++i){
			for (size_t j = i + 1; j < K;++j){
				float Dis = DescDist(test->next[i]->data.CenterDesc, test->next[j]->data.CenterDesc);
				if (Dis < Min) Min = Dis;
				else if (Dis>Max) Max = Dis;
			}
		}
		test = test->next[MinIndex];
	}
	DisThreshold = (Max + Min) / 2.0;
	//DisThreshold = Max;

	for (size_t i = 0; i < test->data.DescIndex.size(); ++i){
		float Distemp = DescDist(test->data.DescIndex[i],a);
		if (Distemp<DisThreshold)
			marker[test->data.DescIndex[i]] += 1;
		else
			marker[test->data.DescIndex[i]] += DisThreshold / Distemp;
	}
}



void VT::Show2Image(Mat & Img1, Mat & Img2){
	Mat img1, img2;
	resize(Img1, img1, Size(0, 0), 1.0*MaxWidth / Img1.cols, 1.0*MaxWidth / Img1.cols, INTER_LINEAR);
	resize(Img2, img2, Size(0, 0), 1.0*MaxWidth / Img2.cols, 1.0*MaxWidth / Img2.cols, INTER_LINEAR);

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



int VT::MatchFeatureNumber(const vector<KeyPoint> &kp1, const vector<KeyPoint> &kp2,const Mat &d1,const Mat &d2, vector<DMatch> &matches) {
	vector<KeyPoint> p1 = kp1;
	vector<KeyPoint> p2 = kp2;
	matches.clear();

	BruteForceMatcher<L2<float>> matcher;//有问题
	//BFMatcher matcher(NORM_L2, true);//true,交叉过滤
	//DescriptorMatcher matcher;
	vector<vector<DMatch>> knnMatches;//保存knn匹配
	const float minRatio = 0.75f;// 1.f / 1.5f;
	const int k = 2;



	clock_t start, finish;
	double totaltime;
	start = clock();



	matcher.knnMatch(d1, d2, knnMatches, k);
	for (size_t i = 0; i < knnMatches.size(); i++) {
		const DMatch& bestMatch = knnMatches[i][0];
		const DMatch& betterMatch = knnMatches[i][1];

		float  distanceRatio = bestMatch.distance / betterMatch.distance;
		if (distanceRatio < minRatio)
			matches.push_back(bestMatch);
	}



	//RANSAC方法计算基础矩阵，并细化匹配结果
	//Align all points
	vector<KeyPoint> alignedKps1, alignedKps2;
	for (size_t i = 0; i < matches.size(); i++) {
		alignedKps1.push_back(p1[matches[i].queryIdx]);
		alignedKps2.push_back(p2[matches[i].trainIdx]);
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
		if (status.data[i] != 0) {
			leftInlier.push_back(alignedKps1[i]);
			rightInlier.push_back(alignedKps2[i]);
			matches[i].trainIdx = index;
			matches[i].queryIdx = index;
			inlierMatch.push_back(matches[i]);
			index++;
		}
	}
	p1 = leftInlier;
	p2 = rightInlier;
	matches = inlierMatch;



	//计算单应矩阵H，并细化匹配结果
	const int minNumbermatchesAllowed = 8;
	if (matches.size() < minNumbermatchesAllowed) {
		return matches.size();
	}


	//Prepare data for findHomography
	vector<Point2f> srcPoints(matches.size());
	vector<Point2f> dstPoints(matches.size());
	for (size_t i = 0; i < matches.size(); i++) {
		srcPoints[i] = p1[matches[i].trainIdx].pt;
		dstPoints[i] = p2[matches[i].queryIdx].pt;
	}

	//find homography matrix and get inliers mask
	vector<uchar> inliersMask(srcPoints.size());
	double reprojectionThreshold = 3.;
	Mat homography = findHomography(srcPoints, dstPoints, CV_FM_RANSAC, reprojectionThreshold, inliersMask);

	vector<DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++) {
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}
	matches.swap(inliers);


	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	//cout << "\nmatches' amount:" << matches.size() << endl;
	//cout << "\nfeatures match takes " << totaltime << " second！" << endl;
	return matches.size();
}



void VT::SearchImage(string file){
	clock_t start, finish;
	double totaltime;
	start = clock();


	ImageData temp;
	temp.file = file;
	Mat Image = imread(file);
	int width = Image.cols;
	int height = Image.rows;
	Mat ResizedImage;
	/*if ((width > height ? width : height)  > MaxWidth){
		//float rate = ResizeRate(width, height);
		//resize(Image, ResizedImage, Size(0, 0), rate, rate, INTER_NEAREST);
		if (width>height)
			resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / width, 1.0*MaxWidth / width, INTER_LINEAR);
		else
			resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / height, 1.0*MaxWidth / height, INTER_LINEAR);
		
		temp.Image = ResizedImage;
	}
	else{
		temp.Image = Image;
	}*/
	temp.Image = ResizeImage(Image);




#if OrbFlag
	ORB extractor(ORBnfeatures);
	Mat descriptor;//临时存放一张图片的描述子
	vector<Desc> ImageDesc;
	extractor(temp.Image, Mat(), temp.FeaturePoint, descriptor);
	for (int k = 0; k < descriptor.rows; ++k){
		Desc DescTemp;
		for (int m = 0; m < D; ++m){
			DescTemp.descriptor[m] = descriptor.at<unsigned char>(k, m);
		}
		ImageDesc.push_back(DescTemp);
	}
#endif
	vector<float> marker(AllDesc.rows);
	vector<float> counter(images.size());
	for (size_t i = 0; i < ImageDesc.size(); ++i){
		SearchDesc(ImageDesc[i].descriptor, marker);
	}
	for (size_t i = 0; i < marker.size(); ++i){
		counter[ImageOfDesc[i]] += marker[i];
	}
	float MaxMarker = 0;
	int MatchedImageIndex = 0;
	for (size_t i = 0; i < counter.size(); ++i){
		counter[i] = counter[i] / images[i].FeaturePoint.size();
		if (counter[i]>MaxMarker){
			MaxMarker = counter[i];
			MatchedImageIndex = i;
		}
	}
	



	//得到最匹配的结果，再进行一次特征匹配，看是否满足
	int DescSize = images[MatchedImageIndex].FeaturePoint.size();
	Mat ResultDesc(DescSize, D, CV_8U);
	int index = 0;
	for (size_t i = 0; i < ImageOfDesc.size(); ++i){
		if (ImageOfDesc[i] == MatchedImageIndex) {
			index = i;

			break;
		}
	}
	for (int i = 0; i < DescSize; ++i){
		for (int j = 0; j < D;++j){
			ResultDesc.at<uchar>(i, j) = AllDesc.at<uchar>(index, j);
		}
		++index;
	}
	vector<DMatch> GoodMatches;
	//BFMatcher matcher(NORM_L2, true);//true,交叉过滤
	//matcher.match(descriptor, ResultDesc, GoodMatches);
	int MatchNum = MatchFeatureNumber(temp.FeaturePoint,images[MatchedImageIndex].FeaturePoint, descriptor, ResultDesc, GoodMatches);
	cout << "MaxDesc,ResultDesc:" << descriptor.rows << "," << GoodMatches.size() << endl;




	cout << "MaxMarker counter:" << MaxMarker << endl;
	cout << "Image to search: " << file << endl;
	cout << "Matched Image" << images[MatchedImageIndex].file << endl;



	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "search the image takes " << totaltime << " seconds！\n" << endl;


	Show2Image(imread(file), imread(images[MatchedImageIndex].file));

}