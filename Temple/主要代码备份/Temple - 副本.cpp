// Temple.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "head.h"
#include "class.h"
#include "VocabularyTree.h"
#include "OrbFeature.h"

int _tmain(int argc, char* argv[])
{
	//测试，打开摄像头
	/*VideoCapture inputVideo;
	if (!inputVideo.open(0)){//0为外部摄像头的ID，1为笔记本内置摄像头的ID
		cout << "USB camera cannot be opened!\n" << endl;
		return 0;
	}
	namedWindow("Video", WINDOW_AUTOSIZE);
	Mat Image;
	while (1){//等到摄像头稳定之后再开始
		inputVideo >> Image;
		imshow("Video", Image);
		waitKey(1);
	}
	inputVideo.release();
	destroyWindow("Video");*/
	

	//测试，读目录下所有文件
	/*string folder = "D:/2016.10.14衢州水亭门/dv";  //此处用的是斜杠，也可以用反斜  
	//但需注意的是由于C语言的特点，要用双反斜杠，即"E:\\MATLAB\\LBP\\scene_categories"  
	//cin >> folder;   //也可以用此段代码直接在DOS窗口输入地址，此时只需正常的单反斜杠即可  
	vector<string> files;
	getFiles(folder, files);  //files为返回的文件名构成的字符串向量组  
	for (int i = 0; i < files.size(); i++) {    //files.size()返回文件数量  
		cout << files[i] << endl;
	}*/


	//生成随机数测试
/*
while (1){
	int k = 3;
	vector<int> random(k);
	for (int i = 0; i < k; ++i){
		while (1){
			srand((unsigned)time(NULL)); //用时间做种，每次产生随机数不一样
			int r1 = rand() % 12;
			srand(r1);
			int r2 = rand() % 12;
			random[i] = r1*r2 / 121.0*50000;
			int flag = 0;//0表示与之前的都不相等
			for (int j = 0; j < i; ++j){
				if (random[i] == random[j]) {
					flag = 1;//一旦与之前的随机数相等就退出，重新生成随机数
					break;
				}
			}
			if (flag == 0) break;
		}
	}
	for (int i = 0; i < k; ++i)
		cout << random[i] << endl;
}
*/


	
	/**/
	VT vt("D:/2016.10.14衢州水亭门/塔data");
	vector<string> files;
	getFiles("D:/2016.10.14衢州水亭门/塔test",files);
	for (size_t i = 0; i < files.size();++i){
		vt.SearchImage(files[i]);
	}
	


	system("pause");
	return 0;
}

