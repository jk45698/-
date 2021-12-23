import os

import cv2
import joblib
from scipy.cluster.vq import *

from sklearn import preprocessing

from pylab import *

testing_path = "/root/work/vision/Dataset/BJTUdataset/newdata1.0/test/"
testing_names = os.listdir(testing_path)
print(testing_names)
image_paths2 = []

for testing_name in testing_names:
    image_path = os.path.join(testing_path, testing_name)
    image_paths2 += [image_path]

topk = [20,40,60]
pkl = "bof2048.pkl"
# Load the classifier, class names, scaler, number of clusters and vocabulary
im_features, image_paths, idf, numWords, voc = joblib.load(pkl)

# %%
# Create feature extraction and keypoint detector objects
detector = cv2.xfeatures2d.SIFT_create()


tongji1 = [{'nm': 0, 'sy': 0, 'tsg': 0, 'sjz': 0, 'tyht': 0}, {'nm': 0, 'sy': 0, 'tsg': 0, 'sjz': 0, 'tyht': 0},
          {'nm': [], 'sy': [], 'tsg': [], 'sjz': [], 'tyht': []}]
tongji2=[{'nm': 0, 'sy': 0, 'tsg': 0, 'sjz': 0, 'tyht': 0}, {'nm': 0, 'sy': 0, 'tsg': 0, 'sjz': 0, 'tyht': 0},
          {'nm': [], 'sy': [], 'tsg': [], 'sjz': [], 'tyht': []}]
tongji3=[{'nm': 0, 'sy': 0, 'tsg': 0, 'sjz': 0, 'tyht': 0}, {'nm': 0, 'sy': 0, 'tsg': 0, 'sjz': 0, 'tyht': 0},
          {'nm': [], 'sy': [], 'tsg': [], 'sjz': [], 'tyht': []}]
states = [tongji1,tongji2,tongji3]

i = 0
for image_path in image_paths2:
    des_list = []

    im = cv2.imread(image_path)
    kpts, des = detector.detectAndCompute(im, None)

    des_list.append((image_path, des))

    # 对文件名分割
    str1 = testing_names[i].split("-")
    print(str1)
    s = str1[0].lower()


    i += 1
    k=0
    for top in topk:
        states[k][1][s] += top
        k+=1

    descriptors = des_list[0][1]

    test_features = np.zeros((1, numWords), "float32")
    words, distance = vq(descriptors, voc)
    for w in words:
        test_features[0][w] += 1

    # Perform Tf-Idf vectorization and L2 normalization
    test_features = test_features * idf
    test_features = preprocessing.normalize(test_features, norm='l2')

    score = np.dot(test_features, im_features.T)
    rank_ID = np.argsort(-score)

    tmp = 0
    k=0
    for top in topk:
        for j, ID in enumerate(rank_ID[0][0:top]):
            s1 = os.path.basename(image_paths[ID]).split("-")
            s2 = s1[0].lower()
            if (s2 == s):
                states[k][0][s]+= 1
                tmp += 1
        states[k][2][s].append(tmp)
        k+=1
        tmp=0

# %%
print(states)
#%%
k=0
for top in topk:
    print(str(top)+":")
    fenzi = 0
    fenmu = 0
    for key in states[k][0]:
        print(key+":"+str(round(float(states[k][0][key])/states[k][1][key],2)))
        fenzi+=states[k][0][key]
        fenmu+=states[k][1][key]
    print("all:"+str(round(float(fenzi/fenmu),2)))
    print()
    print()
    k+=1
