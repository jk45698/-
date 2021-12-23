### 项目说明

本项目包含了交大寻迹任务的所有代码

由于空间限制，数据集单独存放在网盘中，地址：

链接：https://pan.baidu.com/s/1_x7pZpWYwa0hjSUgrlmGjg    提取码：ge98 

### 目录说明

**Dataset**

存放数据集

**mission1**

mission 1文件夹包含了图像检索的所有内容，由于SIFT特征和聚类结果很大，所以只保留了文件夹。最终构建的词袋文件bofxxx.pkl 存放在mission1 目录下，xxx表示聚类中心数，范围：{64, 12, 256, 512, 1024, 2048}

修改search.py 文件的图像路径和pkl文件，运行即可实现检索。

**mission2**

mission 2 文件夹包含了yolo模型，并存储了权重文件。

**warning:**

在上传过程中，系统忽略了 mission2/yolov5-master/runs 文件夹，因此我将其传入百度网盘作为补充：

链接：https://pan.baidu.com/s/1i3tsGnKVXv7giTBZMN8viA 
提取码：w7ne

实现文字检索，使用以下命令：

```
python detect.py --weights ../yolov5-master/runs/train/exp/weights/best.pt --source (you image path)
```

子目录YOLOv5mAP_Bubble_Islands-main 实现了结果的评估。

**result （重点）** 

result目录存储了mission 1 和 mission 2 的运行结果。

**utils**

存放一些小工具，如coco转yolo格式、图像格式转换、文件摘选等操作。



### 写在最后

本项目包含了任务的所有代码。

由于时间紧张，部分代码不规范，可能导致无法从头复现任务，还望读者自行修改。

感谢！



