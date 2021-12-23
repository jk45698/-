import os
import cv2
import joblib
from scipy.cluster.vq import *
from sklearn import preprocessing
from pylab import *

testing_path = ['../Dataset/BJTUdataset/newdata1.0/test', '../Dataset/BJTUdataset/newdata2.0/test']
testing_names = os.listdir(testing_path[0])
#testing_names += os.listdir(testing_path[1])
image_paths2 = []

for testing_name in testing_names:
    image_path = os.path.join(testing_path[0], testing_name)
    image_paths2.append(image_path)

K = [20, 40, 60]
PKL = ['bof64.pkl', 'bof128.pkl', 'bof256.pkl', 'bof512.pkl', 'bof1024.pkl', 'bof2048.pkl']

for pkl in PKL:

    # Load the classifier, class names, scaler, number of clusters and vocabulary
    im_features, image_paths, idf, numWords, voc = joblib.load(pkl)

    print("using " + pkl)

    SIFT_detector = cv2.xfeatures2d.SIFT_create()

    NMrate = {20: [], 40 : [], 60 : []}
    TSGrate = {20: [], 40 : [], 60 : []}
    TYHTrate = {20: [], 40 : [], 60 : []}
    SYrate = {20: [], 40 : [], 60 : []}
    SJZrate = {20: [], 40: [], 60 :[]}
    Location = {'nm': NMrate, 'tsg': TSGrate, 'tyht': TYHTrate, 'sjz' : SJZrate, 'sy': SYrate}
    Sumrate = {20: [], 40: [], 60 :[]}


    i = 0

    # 测试集逐图片计算
    for image_path in image_paths2:
        des_list = []
        img = cv2.imread(image_path)
        newimg = cv2.pyrDown(img) # 图像下采样，降低特征数
        kpts, des = SIFT_detector.detectAndCompute(newimg, None)

        des_list.append((image_path, des))
        
        str = testing_names[i].split("-")
        # s 为当前测试集类别
        s = str[0].lower()
        i += 1
        descriptors = des_list[0][1]

        test_features = np.zeros((1, numWords), "float32")
        words, distance = vq(descriptors, voc)
        for w in words:
            test_features[0][w] += 1

        # Perform Tf-Idf vectorization and L2 normalization
        test_features = test_features * idf
        test_features = preprocessing.normalize(test_features, norm='l2')

        # 内积计算相似度
        score = np.dot(test_features, im_features.T)

        # 按降序排序
        rank_ID = np.argsort(-score)

        courentRate = Location[s]

        for k in K:
            right = 0
            for i, ID in enumerate(rank_ID[0][0:k]):
                s1 = os.path.basename(image_paths[ID]).split("-")[0].lower()
                if(s1 == s):
                    right += 1
                print(s1)
                r = right / k
                
                courentRate[k].append(r)
                Sumrate[k].append(r)
    
    print(Location)