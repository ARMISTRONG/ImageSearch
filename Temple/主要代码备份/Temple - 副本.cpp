// Temple.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
#include "head.h"
#include "class.h"
#include "VocabularyTree.h"
#include "OrbFeature.h"

int _tmain(int argc, char* argv[])
{
	//���ԣ�������ͷ
	/*VideoCapture inputVideo;
	if (!inputVideo.open(0)){//0Ϊ�ⲿ����ͷ��ID��1Ϊ�ʼǱ���������ͷ��ID
		cout << "USB camera cannot be opened!\n" << endl;
		return 0;
	}
	namedWindow("Video", WINDOW_AUTOSIZE);
	Mat Image;
	while (1){//�ȵ�����ͷ�ȶ�֮���ٿ�ʼ
		inputVideo >> Image;
		imshow("Video", Image);
		waitKey(1);
	}
	inputVideo.release();
	destroyWindow("Video");*/
	

	//���ԣ���Ŀ¼�������ļ�
	/*string folder = "D:/2016.10.14����ˮͤ��/dv";  //�˴��õ���б�ܣ�Ҳ�����÷�б  
	//����ע���������C���Ե��ص㣬Ҫ��˫��б�ܣ���"E:\\MATLAB\\LBP\\scene_categories"  
	//cin >> folder;   //Ҳ�����ô˶δ���ֱ����DOS���������ַ����ʱֻ�������ĵ���б�ܼ���  
	vector<string> files;
	getFiles(folder, files);  //filesΪ���ص��ļ������ɵ��ַ���������  
	for (int i = 0; i < files.size(); i++) {    //files.size()�����ļ�����  
		cout << files[i] << endl;
	}*/


	//�������������
/*
while (1){
	int k = 3;
	vector<int> random(k);
	for (int i = 0; i < k; ++i){
		while (1){
			srand((unsigned)time(NULL)); //��ʱ�����֣�ÿ�β����������һ��
			int r1 = rand() % 12;
			srand(r1);
			int r2 = rand() % 12;
			random[i] = r1*r2 / 121.0*50000;
			int flag = 0;//0��ʾ��֮ǰ�Ķ������
			for (int j = 0; j < i; ++j){
				if (random[i] == random[j]) {
					flag = 1;//һ����֮ǰ���������Ⱦ��˳����������������
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
	VT vt("D:/2016.10.14����ˮͤ��/��data");
	vector<string> files;
	getFiles("D:/2016.10.14����ˮͤ��/��test",files);
	for (size_t i = 0; i < files.size();++i){
		vt.SearchImage(files[i]);
	}
	


	system("pause");
	return 0;
}

