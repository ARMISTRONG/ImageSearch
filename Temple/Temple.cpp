// Temple.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "head.h"
#include "class.h"
#include "VocabularyTree.h"
#include "OrbFeature.h"













#include <iostream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cmath>

using namespace std;


void NextStep(int num, int i0, int j0, int in, int jn, int step, const vector<vector<int>>& data, vector<pair<int, int>>& path, vector<pair<int, int>> & bestPath, int score, int &maxScore){
	if (i0 == in && j0 == jn){
		if (score > maxScore){
			bestPath = path;
			maxScore = score;
		}
		else if (score == maxScore){
			if (path.size() < bestPath.size()){
				bestPath = path;
			}
		}
		return;
	}
	else if (step == 0){
		return;
	}
	if (i0>1){//up
		pair<int, int> temp(i0 - 1, j0);
		path.push_back(temp);
		NextStep(num, i0 - 1, j0, in, jn, step - 1, data, path, bestPath, score + data[i0 - 1][j0], maxScore);
		path.pop_back();
	}
	if (i0<num){//down
		pair<int, int> temp(i0 + 1, j0);
		path.push_back(temp);
		NextStep(num, i0 + 1, j0, in, jn, step - 1, data, path, bestPath, score + data[i0 + 1][j0], maxScore);
		path.pop_back();
	}
	if (j0>1){//left
		pair<int, int> temp(i0, j0 - 1);
		path.push_back(temp);
		NextStep(num, i0, j0 - 1, in, jn, step - 1, data, path, bestPath, score + data[i0][j0 - 1], maxScore);
		path.pop_back();
	}
	if (j0<num){//right
		pair<int, int> temp(i0, j0 + 1);
		path.push_back(temp);
		NextStep(num, i0, j0 + 1, in, jn, step - 1, data, path, bestPath, score + data[i0][j0 + 1], maxScore);
		path.pop_back();
	}
}



int FindPath(int num, int i0, int j0, int in, int jn, int step, vector<vector<int>>& data, vector<pair<int, int>>& path){
	int maxScore = 0;
	vector<pair<int, int>> tempPath;
	NextStep(num, i0, j0, in, jn, step, data, tempPath, path, data[i0][j0], maxScore);
	return maxScore;
}

void testFindPath(){
	int a[10][10] = {
		84, 54, 4, 9, 65, 13, 98, 78, 3, 14,
		4, 56, 45, 4, 53, 45, 45, 4, 78, 47,
		78, 54, 32, 75, 78, 2, 58, 21, 3, 6,
		78, 45, 46, 8, 54, 45, 78, 23, 95, 4,
		78, 96, 7, 56, 74, 8, 6, 46, 6, 46,
		45, 74, 97, 2, 54, 26, 3, 46, 36, 1,
		78, 5, 5, 79, 36, 64, 8, 63, 6, 5,
		56, 72, 5, 5, 54, 45, 31, 13, 46, 1,
		87, 8, 6, 3, 58, 12, 48, 12, 12, 4,
		35, 98, 54, 54, 89, 16, 13, 54, 12, 41
	};
	vector<vector<int>> data(10, vector<int>(10, 0));
	for (int i = 0; i < 10; ++i){
		for (int j = 0; j < 10; ++j){
			data[i][j] = a[i][j];
		}
	}
	vector<pair<int, int>> path;
	int maxScore = FindPath(9, 0, 0, 9, 9, 20, data, path);
	cout << "maxScore:" << maxScore << endl;
	for (int i = 0; i < path.size(); ++i){
		cout << i + 1 << ": " << path[i].first << "," << path[i].second << endl;
	}
}










string n = "..........##....#..#.#...#..#..#..#..#...#.#..#....##..........";//7*9
string t = "..........#######.....#........#........#........#.............";
string e = "..........#######..#........#######..#........#######..........";
string s = "..........#######..#........#######........#..#######..........";

string nMask = "####..########.#################################.########..####";//7*9
string tMask = "###########################...###......###......###......###...";
string eMask = "###############################################################";
string sMask = "###############################################################";

string init = "...............................................................";




void getData(int &N, int &M, vector<vector <char> >&data){
	cin >> N >> M;
	data.clear();
	for (int i = 0; i<N + 2; ++i){
		vector<char> a(M + 2, '.');
		data.push_back(a);
	}
	for (int i = 1; i<N + 1; ++i){
		for (int j = 1; j<M + 1; ++j){
			cin >> data[i][j];
		}
	}
}


int match(vector<vector <char> >&data, int I, int J, string test, string mask){
	string a(init);
	for (int i = I, count = 0; i<I + 7; ++i){
		for (int j = J; j<J + 9; ++j){
			if (mask[count] == '#') a[count] = data[i][j];
			else a[count] = '.';
			++count;
		}
	}
	if (a == test)//写错变量
		return 1;
	else
		return 0;
}


void rotate(vector<vector <char> >&src, int N, int M, vector<vector <char> >&dst){
	dst.clear();
	for (int i = 0; i<M; ++i){
		vector<char> a(N);
		dst.push_back(a);
	}
	for (int i = 0; i<M; ++i){
		for (int j = 1; j<N; ++j){
			dst[i][j] = src[N-1-j][i];//旋转的时候出问题
		}
	}
}


int search(vector<vector <char> >&data, string test, string mask){
	int count = 0;
	for (int i = 0; i + 7 <= data.size(); ++i){
		for (int j = 0; j + 9 <= data[0].size(); ++j){
			count += match(data, i, j, test, mask);
		}
	}
	return count;
}


void testMethod(){
	int N, M;
	vector<vector <char> > data;
	getData(N, M, data);
	vector<vector <char> > dataRotate = data;


	char NTES[4] = { 'N', 'T', 'E', 'S' };
	int result[4] = { 0 };
	result[0] += search(data, n, nMask);
	result[1] += search(data, t, tMask);
	result[2] += search(data, e, eMask);
	result[3] += search(data, s, sMask);
	for (int i = 0; i<3; ++i){
		vector<vector <char> > dataRotateTemp;
		rotate(dataRotate, dataRotate.size(), dataRotate[0].size(), dataRotateTemp);
		result[0] += search(dataRotateTemp, n, nMask);
		result[1] += search(dataRotateTemp, t, tMask);
		result[2] += search(dataRotateTemp, e, eMask);
		result[3] += search(dataRotateTemp, s, sMask);
		dataRotate = dataRotateTemp;
	}
	for (int i = 0; i<4; ++i){
		cout << NTES[i] << ": " << result[i] << endl;
	}
}







string countAndSay(int n) {
	string result = "1";
	if (n <= 1) {
		return result;
	}
	for (int i = 1; i<n; ++i){
		string temp = "";
		char current = result[0];
		int count = 0;
		for (int j = 0; j<result.size(); ++j){
			if (result[j] == current){
				++count;
			}
			else if (   result[j] != current ){
				stringstream ss;
				string s;
				ss << count;
				ss >> s;
				temp = temp + s + current;
				
				current = result[j];
				count = 1;
			}
			if (j == result.size() - 1){
				stringstream ss;
				string s;
				ss << count;
				ss >> s;
				temp = temp + s + current;
			}
		}
		result = temp;
	}

	return result;
}



int reverse(int a){
	int remain = a;
	vector<int> A;
	while (remain>0){
		A.push_back(remain % 10);
		remain /= 10;
	}
	int result = 0;
	for (int i = A.size() - 1, step = 1; i >= 0; --i, step *= 10){
		result += step*A[i];
	}
	return result;
}


int reverseAdd(int a, int b){

	/*int a, b;
	char c;
	cin >> a >> c >> b;
	cout << reverseAdd(a, b) << endl;*/

	if (a<1 || a>70000) return -1;
	if (b<1 || b>70000) return -1;
	int A = reverse(a);
	int B = reverse(b);
	int C = A + B;
	cout << C << endl;
	return C;
}


void rotate(char c, int *a){
	int temp;
	switch (c){
	case 'L':
		temp = a[1];
		a[1] = a[5];
		a[5] = a[2];
		a[2] = a[6];
		a[6] = temp;
		break;
	case 'R':
		temp = a[1];
		a[1] = a[6];
		a[6] = a[2];
		a[2] = a[5];
		a[5] = temp;
		break;
	case 'F':
		temp = a[3];
		a[3] = a[5];
		a[5] = a[4];
		a[4] = a[6];
		a[6] = temp;
		break;
	case 'B':
		temp = a[3];
		a[3] = a[6];
		a[6] = a[4];
		a[4] = a[5];
		a[5] = temp;
		break;
	case 'A':
		temp = a[3];
		a[3] = a[1];
		a[1] = a[4];
		a[4] = a[2];
		a[2] = temp;
		break;
	case 'C':
		temp = a[3];
		a[3] = a[2];
		a[2] = a[4];
		a[4] = a[1];
		a[1] = temp;
		break;
	}
}

void getResult(string s, int *a){
	/*int a[7] = { 0, 1, 2, 3, 4, 5, 6 };
	string s;
	cin >> s;
	getResult(s, a);
	for (int i = 1; i <= 6; ++i){
	cout << a[i];
	}*/
	for (int i = 0; i<s.size(); ++i){
		rotate(s[i], a);
	}
}











#define M 1000
int dist[6][6] = { 0, 2, 10, 5, 3, M,
M, 0, 12, M, M, 10,
M, M, 0, M, 7, M,
2, M, M, 0, 2, M,
4, M, M, 1, 0, M,
3, M, 1, M, 2, 0 };


void Floyd(int  path[6][6], int  A[6][6]){
	for (int k = 0; k<6; ++k){
		for (int i = 0; i<6; ++i){
			for (int j = 0; j<6; ++j){
				if (A[i][j]>A[i][k] + A[k][j]){
					A[i][j] = A[i][k] + A[k][j];
					path[i][j] = k;
				}
			}
		}
	}
}


void outputPath(int  path[6][6], int s, int e){
	if (path[s][e] == -1) return;
	else{
		outputPath(path, s, path[s][e]);
		cout << "," << path[s][e] + 1;
	}
}


void getPath(int X, int Y){
	/*int X, Y;
	cin >> X >> Y;
	getPath(X, Y);*/
	int path[6][6] = { -1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1,
		-1, -1, -1, -1, -1, -1 };
	int A[6][6] = { 0 };
	for (int i = 0; i<6; ++i){
		for (int j = 0; j<6; ++j){
			if (i != j){
				if (i == Y - 1 || j == Y - 1)
					A[i][j] = M;
				else
					A[i][j] = dist[i][j];
			}
		}
	}

	Floyd(path, A);
	if (A[5 - 1][X - 1] >= M) {
		cout << 1000 << endl;
		cout << "[]";
	}
	else{
		cout << A[5 - 1][X - 1] << endl;
		cout << "["<<5;
		outputPath(path, 5 - 1, X - 1);
		cout <<","<< X  << "]";
	}

}






int Jisuan(int a, char b, char c){
	/*string s;
	int result = 0;
	cin >> s;
	if (s.size() == 0){
	cout << result << endl;
	return 0;
	}
	else if (s.size() == 1){
	result += s[0] - '0';
	return 0;
	}

	result = s[0] - '0';
	for (int i = 1; i<s.size(); i += 2){
	result = Jisuan(result, s[i], s[i + 1]);
	}
	cout << result << endl;*/
	int result = 0;
	switch (b){
	case '+':
		result = a + (c - '0');
		break;
	case '-':
		result = a - (c - '0');
		break;
	case '*':
		result = a*(c - '0');
		break;
	}
	return result;
}


void exist(vector<double> & result, double a){
	if (result.size() == 0) {
		result.push_back(a);
		return;
	}
	vector<double>::iterator b = result.begin();
	for (int i = 0; i<result.size(); ++b, ++i){
		if (a == result[i]) return;
		else if (a>result[i]) {
			result.insert(b, a);
			return;
		}
	}
	result.push_back(a);
	return;
}


void Find(vector<double> & result, int w, int x, int y, int z){
	/*int w, x, y, z;
	cin >> w >> x >> y >> z;
	vector<double> result;
	Find(result, w, x, y, z);
	cout << result.size() << endl;
	for (int i = 0; i < result.size(); ++i){
	cout << result[i] << endl;
	}*/
	result.clear();
	for (int i = w; i <= x; ++i){
		for (int j = y; j <= z; ++j){
			exist(result, (1.0*i) / j);
		}
	}

}



void beibao(){
	int n;
	vector<int> data;
	long int capacity = 0;
	cin >> n;
	for (int i = 0; i < n;++i){
		int temp;
		cin >> temp;
		data.push_back(temp);
		capacity += temp;
	}
	capacity /= 2;

	vector<vector<long int>> V(n+1,vector<long int> (capacity+1,0));
	for (int i = 1; i <= n; i++){
		for (int j = 1; j <= capacity; j++){
			if (j<data[i - 1])
				V[i][j] = V[i - 1][j];
			else
				V[i][j] = max(V[i - 1][j], V[i - 1][j - data[i - 1]] + data[i - 1]);
		}
	}

	cout << "Dynamic Matrix: " << endl;
	for (int i = 1; i <= n; i++){
		for (int j = 1; j <= capacity; j++){
			cout << V[i][j] << " ";
		}
		cout << endl;
	}

	int j = capacity;
	for (int i = n; i>0; i--){
		if (V[i][j]>V[i - 1][j]){
			j = j - data[i - 1];
			cout << data[i-1] << " ";
		}
	}
	cout << endl;
}





int _tmain(int argc, char* argv[])
{
	

	beibao();
	





	
	




	/*
	//可以运行
	cvNamedWindow("test", CV_WINDOW_AUTOSIZE);
	CvCapture* capture = cvCreateCameraCapture(0);
	assert(capture != NULL);
	IplImage* frame;
	while (1){
		frame = cvQueryFrame(capture);
		double timestamp = cvGetCaptureProperty(capture, CV_CAP_PROP_POS_MSEC);
		if (!frame) break;
		cout << timestamp;
		cvShowImage("test", frame);
		char c = cvWaitKey(33);
		if (c == 27) break;
	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("test");



	//打开摄像头
	cv::VideoCapture inputVideo;
	//if (!inputVideo.open("C:\\Users\\Administrator\\Desktop\\CameraTest\\CameraTest\\Results\\video0.avi")){//打开视频文件
	if (!inputVideo.open(0)){//0为外部摄像头的ID，1为笔记本内置摄像头的ID
	cout << "USB camera cannot be opened!\n" << endl;
	return 0;
	}
	//摄像头打开之后才能设置
	//inputVideo.set(CV_CAP_PROP_FPS, Cam_FPS);//调整帧率
	//inputVideo.set(CV_CAP_PROP_EXPOSURE, 0.5);
	//inputVideo.set(CV_CAP_PROP_CONTRAST, 0.5);
	//inputVideo.set(CV_CAP_PROP_FRAME_WIDTH, 1024);
	//inputVideo.set(CV_CAP_PROP_FRAME_HEIGHT, 768);
	//获得视频的宽高
	int Width = inputVideo.get(CV_CAP_PROP_FRAME_WIDTH);
	int Height = inputVideo.get(CV_CAP_PROP_FRAME_HEIGHT);
	//cout << "(w,h):" << Width << "," << Height << endl;
	//cout << "get FPS:" << inputVideo.get(CV_CAP_PROP_FPS) << endl;
	cv::namedWindow("Video", cv::WINDOW_AUTOSIZE);
	//输出视频
	cv::VideoWriter video("./Results/video.avi", CV_FOURCC('P', 'I', 'M', '1'), 20, cvSize(Width, Height), 0);//输出彩色或者灰度要改参数

	Mat Image;
	while (1){//等到摄像头稳定之后再开始
	inputVideo >> Image;
	imshow("Video", Image);
	cv::waitKey(1);
	}
	*/




	/*
	
	
	
	VT vt("E:/2016.10.14衢州水亭门/塔data");
	vector<string> files;
	getFiles("E:/2016.10.14衢州水亭门/塔test",files);
	for (size_t i = 0; i < files.size();++i){
		vt.SearchImage(files[i]);
	}
	
	
	*/


	
	



	system("pause");
	return 0;
}

