'''
实现功能： 将train set 所有图像的 SIFT feature 提取出来
保存到SIFT_Features文件夹中
'''

import cv2
import numpy as np
import os

TrainSetPath = ["../Dataset/BJTUdataset/newdata1.0/train", 
                "../Dataset/BJTUdataset/newdata2.0/train", 
                "../Dataset/BJTUdataset/olddata/train"]

IMGPath = []

for i in range(3):

    FileList = os.listdir(TrainSetPath[i])

    for file in FileList:

        if (not "json" in file):
            IMGPath.append(os.path.join(TrainSetPath[i], file))

print("number of train data :", len(IMGPath))

FeatureList = []
descriptors = []
Badpath = []

SIFT_detector = cv2.xfeatures2d.SIFT_create()

for i, imgpath in enumerate(IMGPath):

    if i % 100 == 0:
        print("processing " + str(i) + "th" + " img in " + str(len(IMGPath)) + " imgs")

    img = cv2.imread(imgpath)
    newimg = cv2.pyrDown(img) # 图像下采样，降低特征数
    kpts, des = SIFT_detector.detectAndCompute(newimg, None)

    # 避免无法提取特征的图像导致的异常
    if des is None:
        des = np.array([])
        Badpath.append(imgpath)

    FeatureList.append((imgpath, des))
    descriptors += des.tolist()

    if (i % 2000 == 0) and (i != 0) :
        FeatureArray = np.array(FeatureList, 'object')
        print("shape of FeatureArray " + str(i) + ": ", FeatureArray.shape)
        np.save("../mission1/SIFT_Feature/features" + str(i), FeatureArray)
        FeatureArray = np.ones(1) # 释放空间

        DescriptorArray = np.array(descriptors, 'float32')
        print("shape of DescriptorArray " + str(i) + ": ", DescriptorArray.shape)
        np.save("../mission1/SIFT_Feature/descriptors" + str(i), DescriptorArray)
        DescriptorArray = np.ones(1)

        FeatureList = []
        descriptors = []


FeatureArray = np.array(FeatureList, 'object')
print("shape of FeatureArray: ", FeatureArray.shape)
np.save("../mission1/SIFT_Feature/features.npy", FeatureArray)
FeatureArray = np.ones(1) # 释放空间

DescriptorArray = np.array(descriptors, 'float32')
print("shape of DescriptorArray: ", DescriptorArray.shape)
np.save("../mission1/SIFT_Feature/descriptors.npy", DescriptorArray)

print(Badpath)