# "author:youngkun date:20180615 function:change the size of pictures in one folder"
import cv2
import os

# 读取不规范图片数据，并规范为相同大小格式

image_size = 100  # 设定尺寸
source_path = "/data/Documents/pcl2022/opencv_cc_tensorflow/golds_and_monkey/validation/monkey_king/"  # 源文件路径
# source_path = "/home/zkpark/Documents/pcl2022/opencv_cc_tensorflow/golds_and_monkey/train/gold/"  # 源文件路径
target_path = "./dataset2/test/"  # 输出目标文件路径
# target_path = "./dataset2/gold/"  # 输出目标文件路径

if not os.path.exists(target_path):
    os.makedirs(target_path)

image_list = os.listdir(source_path)  # 获得文件名

i = 0
for file in image_list:
    i = i + 1
    image_source = cv2.imread(source_path + file)  # 读取图片
    print(image_source)
    image = cv2.resize(image_source, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)  # 修改尺寸
    cv2.imwrite(target_path + '1-' + str(i) + ".jpg", image)  # 重命名并且保存
print("批量处理完成")


