#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>			//包含dnn模块的头文件
#include <iostream>
#include <fstream>				//文件流进行txt文件读取

using namespace cv;
using namespace cv::dnn;			//包含dnn的命名空间
using namespace std;

//定义名称，用于后面的显示操作
String objNames[] = { "glod","monkey_king"};
int main() {

	//模型的pb文件
	string bin_model = "/data/Documents/pcl2022/opencv_cc_tensorflow/Neural_network/TFCNN/TFV2_MOLDEL/classify_gold.pb";

	// load DNN model
	Net net = readNetFromTensorflow(bin_model);

	//获取各层信息
	// vector<string> layer_names = net.getLayerNames();		//此时我们就可以获取所有层的名称了，有了这些可以将其ID取出
	// for (int i = 0; i < layer_names.size(); i++) {
	// 	int id = net.getLayerId(layer_names[i]);			//通过name获取其id
	// 	auto layer = net.getLayer(id);						//通过id获取layer
	// 	printf("layer id:%d,type:%s,name:%s\n", id, layer->type.c_str(), layer->name.c_str());	//将每一层的id，类型，姓名打印出来（可以明白此网络有哪些结构信息了）
	// }

	Mat src = imread("/data/Documents/pcl2022/opencv_cc_tensorflow/Neural_network/TFCNN/dataset2/monkey_king/1-13.jpg");	
															//2-20,2-21,2-11,2-1
	if (src.empty()) {
		cout << "could not load image.." << endl;
		getchar();
		return -1;
	}
	imshow("src", src);

	//构建输入(根据建立的网络模型时的输入)
	Mat inputBlob = blobFromImage(src, 1.0, Size(100, 100), Scalar(), true, false);	//我们要将图像resize成100*100的才是我们神经网络可以接受的宽高
	//参数1：输入图像，参数2：默认1.0表示0-255范围的，参数3：设置输出的大小，参数4：均值对所有数据中心化预处理,参数5：是否进行通道转换(需要),参数6：，参数7：默认深度为浮点型

	//上方得到的inputBlob是4维的（在变量窗口看dim），所以在imagewatch中无法查看

	//设置输入
	//现在要将其输入到创建的网络中
	net.setInput(inputBlob);

	//进行推断得到输出
	//让网络执行得到output,调用forward可以得到一个结果
	//此处不给参数，得到的是最后一层的结果，也可以输入层数得到任何一层的输出结果
	Mat probMat = net.forward();	//通过前面的输出层看最后一层，可以知道输出7个分类
	
	//对数据进行序列化（变成1行n列的，可以在后面进行方便的知道是哪个index了）
	Mat prob = probMat.reshape(1, 1);		//reshape函数可以进行序列化，（输出为1通道1行的数据，参数1:1个通道，参数2:1行）将输出结果变成1行n列的，但前面probMat本身就是7*1*1
	//										//实际结果probMat和prob相同
	//										//当其他网络probMat需要序列化的时候，reshape就可以了

	//此时找到最大的那个
	Point classNum;
	double classProb;
	minMaxLoc(prob, NULL, &classProb, NULL, &classNum);//此时只获取最大值及最大值位置，最小值不管他
	int index = classNum.x;		//此时得到的是最大值的列坐标。就是其类的索引值，就可以知道其类名了
	printf("\n current index=%d,possible:%2f,name=%s\n", index, classProb, objNames[index].c_str());	
	//此时可以将名称打印到图片上去
	putText(src, objNames[index].c_str(), Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2, 8);
	imshow("result", src);

	waitKey(50000);
	return 0;
}
// g++ --std=c++11 `pkg-config opencv --cflags` inference_cv.cc  -o result `pkg-config opencv --libs` && ./result   