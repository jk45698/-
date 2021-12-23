'''
将图像文件统一成jpg后缀
在 Image_format_Process.ipynb 处理之后
数据都已经是 jpg类型， 但是有些数据由于本身是jpg类型，但后缀被作者强改为png等
这时，该问题将被忽略
所以，经过上一步处理之后，仍要修改图像后缀名
'''

import os

Path = "./Dataset/BJTUdataset/newdata2.0/train"

FileList = os.listdir(Path)

for file in FileList:
    if not file.endswith(".json"):
        frontname = os.path.splitext(file)[0]
        frontname += '.jpg'
        print(frontname)
        os.rename(os.path.join(Path, file), os.path.join(Path, frontname))