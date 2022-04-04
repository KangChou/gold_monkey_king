#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>			//包含dnn模块的头文件
#include <iostream>
#include <fstream>				//文件流进行txt文件读取

using namespace cv;
using namespace cv::dnn;			//包含dnn的命名空间
using namespace std;

int main() {
	//车辆分类，输入模型的地址
	string bin_model = "./classify_gold.pb";

	//加载模型
	Net net = readNetFromTensorflow(bin_model);

	//获取各层信息
	vector<string> layer_names = net.getLayerNames();		//此时我们就可以获取所有层的名称了，有了这些可以将其ID取出
	for (int i = 0; i < layer_names.size(); i++) {
		int id = net.getLayerId(layer_names[i]);			//通过name获取其id
		auto layer = net.getLayer(id);						//通过id获取layer
		printf("layer id:%d,type:%s,name:%s\n", id, layer->type.c_str(), layer->name.c_str());	//将每一层的id，类型，姓名打印出来（可以明白此网络有哪些结构信息了）
	}
	waitKey(0);
	return 0;
}
// g++ --std=c++11 `pkg-config opencv --cflags` main.cc  -o demo `pkg-config opencv --libs` && ./demo   