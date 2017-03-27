#include "stdafx.h"
#include "VocabularyTree.h"

VT::VT(const string & path){
	//cvNamedWindow("video", 1);
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
			//resize(Image, ResizedImage, Size(MaxWidth, MaxWidth*1.0 / Image.size().width*Image.size().height));
			if (width>height)
				resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / width, 1.0*MaxWidth / width);
			else
				resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / height, 1.0*MaxWidth / height);

			temp.Image = ResizedImage;
			//imshow("video", ResizedImage);
		}
		else{
			temp.Image = Image;
			//imshow("video",temp.Image);
		}
		//waitKey(1);



#if OrbFlag

		ORB extractor;
		Mat descriptor;//临时存放一张图片的描述子,descriptor.type():0:CV_8U
		extractor(temp.Image, Mat(),temp.FeaturePoint, descriptor);
		for (int k = 0; k < descriptor.rows; ++k){
			Desc DescTemp;
			for (int m = 0; m < D; ++m){
				DescTemp.descriptor[m] = descriptor.at<unsigned char>(k, m);
			}
			DescTemp.ImageIndex = i;
			descriptors.push_back(DescTemp);
		}
		temp.FeatureCounter = descriptor.rows;
		images.push_back(temp);
		//cout << temp.file << endl;
	}

#else
		/* 
		SIFT sift; 
		sift(img_1, Mat(), keyPoints_1, descriptors_1); 
		sift(img_2, Mat(), keyPoints_2, descriptors_2); 
		BruteForceMatcher<L2<float> >  matcher; 
		*/  

		/* 
		SURF surf; 
		surf(img_1, Mat(), keyPoints_1); 
		surf(img_2, Mat(), keyPoints_2); 
		SurfDescriptorExtractor extrator; 
		extrator.compute(img_1, keyPoints_1, descriptors_1); 
		extrator.compute(img_2, keyPoints_2, descriptors_2); 
		BruteForceMatcher<L2<float> >  matcher; 
		*/ 
#if FeatureFlag
		//SIFT( int nfeatures=0, int nOctaveLayers=3,double contrastThreshold = 0.04, double edgeThreshold = 10,double sigma = 1.6)
		//nfeatures：特征点数目（算法对检测出的特征点排名，返回最好的nfeatures个特征点）。
		//nOctaveLayers：金字塔中每组的层数（算法中会自己计算这个值，后面会介绍）。
		//contrastThreshold：过滤掉较差的特征点的对阈值。contrastThreshold越大，返回的特征点越少。
		//edgeThreshold：过滤掉边缘效应的阈值。edgeThreshold越大，特征点越多（被多滤掉的越少）。
		//sigma：金字塔第0层图像高斯滤波系数，也就是σ。
		SiftFeatureDetector featuredtc(0,3,0.06,6,1.6);
		SiftDescriptorExtractor extractor;
#else
		//SURF(double hessianThreshold,int nOctaves = 4, int nOctaveLayers = 2,bool extended = true, bool upright = false);
		//为了判断一个点是否是surf特征点，需要在这个点周围构造一个hessian矩阵。决定这个矩阵的因素是hessian，它影响着特征点的鲁棒性。
		//minHessian是一个阈值，它决定了那些值是你接受的关键点。使用的时候，minHessian值越高，得到的关键点越少，但是关键点也就跟好。
		//如果minHessian越小，得到的关键点会更多，但是关键点的质量也就不高。一般，minHessian的值在400 和 800 之间。
		SurfFeatureDetector featuredtc(500);
		SurfDescriptorExtractor extractor;
#endif
		//clock_t start, finish;
		//double totaltime;
		//start = clock();
		featuredtc.detect(temp.Image, temp.FeaturePoint);
		Mat descriptor;//临时存放一张图片的描述子,descriptor.type():5:CV_32F
		extractor.compute(temp.Image, temp.FeaturePoint, descriptor);
		//cout << "descriptor counter:" << descriptor.rows << endl;
		//finish = clock();
		//totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
		//cout << "\nfeatures extract takes " << totaltime << " second!" << endl;


		for (int k = 0; k < descriptor.rows; ++k){
			Desc DescTemp;
			for (int m = 0; m < D; ++m){
				DescTemp.descriptor[m] = descriptor.at<float>(k, m);
			}
			DescTemp.ImageIndex = i;
			descriptors.push_back(DescTemp);
		}
		temp.FeatureCounter = descriptor.rows;
		images.push_back(temp);
		cout << temp.file << endl;
	}
#endif
		
	cout << images.size() << " images are read!" << endl;
	cout << descriptors.size() << " descriptors are created!\n" << endl;
	//destroyWindow("video");




	//计时
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "\nfeatures extract takes " << totaltime << " seconds!" << endl;




	{
	root = new Vtree;
	for (size_t i = 0; i < descriptors.size();++i){
		root->data.DescIndex.push_back(i);
	}
	clock_t start, finish;
	double totaltime;
	start = clock();
	cout << "start to create Vtree!" << endl;
	CreateVtree(root);
	cout << "Vtree created!" << endl;
	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "building the vacabulary tree takes " << totaltime << " seconds！" << endl;
	}

}


void VT::CoutDesc(float a[]){
	for (int i = 0; i < D; ++i){
		cout << a[i] << " ";
	}
	cout << endl;
}


float VT::DescDistance(float a[], float b[]){
	float dis = 0;
	for (int i = 0; i < D; ++i){
		if (a[i] == b[i]) continue;
		dis += pow(a[i] - b[i], 2);
	}
	dis = pow(dis, 0.5);
	return dis;
}


void VT::DescMean(vector<VtreeData> &input){
	int size0 = input.size();
	for (int i = 0; i < size0; ++i){
		for (int j = 0; j < D; ++j){
			double mean = 0;
			int size1 = input[i].DescIndex.size();
			for (int k = 0; k < size1; ++k){
				mean += descriptors[input[i].DescIndex[k]].descriptor[j];
			}
			input[i].CenterDesc[j] = mean / input[i].DescIndex.size();
		}
	}
}


int VT::Nearest(int InputIndex, vector<VtreeData> cluster){
	float MinDis = 1000000;
	int ResultIndex = 0;
	int size = cluster.size();
	for (int i = 0; i < size; ++i){
		float dis = DescDistance(descriptors[InputIndex].descriptor, cluster[i].CenterDesc);
		if (dis < MinDis){
			ResultIndex = i;
			MinDis = dis;
		}
	}
	return ResultIndex;
}


vector<VtreeData> VT::K_Means(int k, VtreeData input){
	vector<VtreeData> result(k);

	vector<int> random(k);
	int size = input.DescIndex.size();
	for (int i = 0; i < k; ++i){//初始化分类中心
		while (1){
			srand((unsigned)time(NULL)); //用时间做种，每次产生随机数不一样
			int r1 = rand() % 12;
			srand(r1);
			int r2 = rand() % 18;
			random[i] = r1*r2 / 187.0 *( size - 1);
			//random[i] = rand() % input.DescIndex.size();  //产生随机数
			int flag = 0;//0表示与之前的都不相等
			for (int j = 0; j < i; ++j){
				if (random[i] == random[j]) {
					flag = 1;//一旦与之前的随机数相等就退出，重新生成随机数
					break;
				}
			}
			if (flag == 0) break;
		}
		for (int m = 0; m < D; ++m){
			result[i].CenterDesc[m] = descriptors[input.DescIndex[random[i]]].descriptor[m];
		}
	}

	
	vector<VtreeData> LastIter(k);
	for (int i = 0; i < MaxIter; ++i){
		float MaxBias = 0;
		for (size_t j = 0; j < LastIter.size(); ++j){
			float tempBias = DescDistance(result[j].CenterDesc, LastIter[j].CenterDesc);
			if (tempBias>MaxBias) MaxBias = tempBias;
		}
		if (MaxBias < Bias) {
			cout << "iteration: " << i << endl;
			break;
		}
		else{
			for (size_t m = 0; m < result.size(); ++m){//每次迭代要先清空一下容器
				result[m].DescIndex.clear();
			}

			for (size_t j = 0; j < input.DescIndex.size(); ++j){
				result[Nearest(input.DescIndex[j], result)].DescIndex.push_back(input.DescIndex[j]);
			}

			LastIter = result;
			DescMean(result);
		}
	}

	return result;
}



#if 0
void VT::CreateVtree(Vtree * root){
	if (root->data.DescIndex.size() <= MinCluster){
		for (int i = 0; i < K; ++i){
			root->next[i] = NULL;
		}
	}
	else{
		vector<VtreeData> result = K_Means(K, root->data);
		for (int i = 0; i < K; ++i){
			root->next[i] = new Vtree;
			root->next[i]->data = result[i];
			cout << result[i].DescIndex.size() << endl;
			CreateVtree(root->next[i]);
		}
	}
	return;
}
#else

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
				points.at<float>(i, j) = descriptors[root->data.DescIndex[i]].descriptor[j];
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
#endif 




void VT::SearchDesc(float a[], vector<float> &marker){
	Vtree * test = root;
	float Min=1000000,Max=0,DisThreshold = 1000000;
	while (test->next[0] != NULL){
		float MinDis = 1000000;
		int MinIndex = 0;
		float DisTemp = 0;
		for (int i = 0; i < K; ++i){
			DisTemp = DescDistance(a, test->next[i]->data.CenterDesc);
			if (DisTemp<MinDis){
				MinDis = DisTemp;
				MinIndex = i;
			}
		}
		for (size_t i = 0; i < K-1; ++i){
			for (size_t j = i + 1; j < K;++j){
				float Dis = DescDistance(test->next[i]->data.CenterDesc, test->next[j]->data.CenterDesc);
				if (Dis < Min) Min = Dis;
				else if (Dis>Max) Max = Dis;
			}
		}
		test = test->next[MinIndex];
	}
	//DisThreshold = (Max + Min) / 2.0;
	DisThreshold = Max;

	for (size_t i = 0; i < test->data.DescIndex.size(); ++i){
		float Distemp = DescDistance(a, descriptors[test->data.DescIndex[i]].descriptor);
		//if (Distemp<DisThreshold)
		//	marker[test->data.DescIndex[i]] += 1;
		//else
			marker[test->data.DescIndex[i]] += DisThreshold / Distemp;
	}
}



void VT::Show2Image(Mat & Img1, Mat & Img2){
	Mat img1, img2;
	resize(Img1, img1, Size(0, 0), 1.0*MaxWidth / Img1.cols, 1.0*MaxWidth / Img1.cols);
	resize(Img2, img2, Size(0, 0), 1.0*MaxWidth / Img2.cols, 1.0*MaxWidth / Img2.cols);

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

	if ((width > height ? width : height)  > MaxWidth){
		//resize(Image, ResizedImage, Size(MaxWidth, MaxWidth*1.0 / Image.size().width*Image.size().height));
		if (width>height)
			resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / width, 1.0*MaxWidth / width);
		else
			resize(Image, ResizedImage, Size(0, 0), 1.0*MaxWidth / height, 1.0*MaxWidth / height);

		temp.Image = ResizedImage;
	}
	else{
		temp.Image = Image;
	}
#if OrbFlag
	ORB extractor;
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
#else

#if FeatureFlag
	SiftFeatureDetector featuredtc(0, 3, 0.06, 6, 1.6);
	SiftDescriptorExtractor extractor;
#else
	SurfFeatureDetector featuredtc(500);
	SurfDescriptorExtractor extractor;
#endif
	Mat descriptor;//临时存放一张图片的描述子
	vector<Desc> ImageDesc;
	featuredtc.detect(temp.Image, temp.FeaturePoint);
	extractor.compute(temp.Image, temp.FeaturePoint, descriptor);

	/*cout << descriptor.type() << endl;//5:CV_32F float类型，但实际只有整数0-255？
	for (int i = 0; i < D;++i){
		setprecision(9);
		cout << descriptor.at<float>(0, i) << " ";
	}*/
	//cout << descriptor << endl;

	for (int k = 0; k < descriptor.rows; ++k){
		Desc DescTemp;
		for (int m = 0; m < D; ++m){
			DescTemp.descriptor[m] = descriptor.at<float>(k, m);
		}
		ImageDesc.push_back(DescTemp);
	}

#endif

	vector<float> marker(descriptors.size());
	vector<float> counter(images.size());
	for (size_t i = 0; i < ImageDesc.size(); ++i){
		SearchDesc(ImageDesc[i].descriptor, marker);
	}
	for (size_t i = 0; i < marker.size(); ++i){
		counter[descriptors[i].ImageIndex] += marker[i];
	}
	float MaxMarker = 0;
	int MatchedImageIndex = 0;
	for (size_t i = 0; i < counter.size(); ++i){
		counter[i] = counter[i] / images[i].FeatureCounter;
		if (counter[i]>MaxMarker){
			MaxMarker = counter[i];
			MatchedImageIndex = i;
		}
	}
	cout << "MaxMarker counter:" << MaxMarker << endl;
	cout << "Image to search: " << file << endl;
	cout << "Matched Image" << images[MatchedImageIndex].file << endl;

	finish = clock();
	totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "search the image takes " << totaltime << " seconds！\n" << endl;

	Show2Image(imread(file), imread(images[MatchedImageIndex].file));


	/*imshow("Image to search",imread(file));
	waitKey(0);
	imshow("Matched Image",imread(images[MatchedImageIndex].file));
	waitKey(0);*/
}