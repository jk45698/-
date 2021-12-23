import os
import numpy as np
from numpy.core.defchararray import center
import joblib
from sklearn import preprocessing

num_cluster = [64, 128, 256, 512, 1024, 2048]
Path1 = "../mission1/kmeans/"
Path2 = "../mission1/SIFT_Feature/"

descriptors = np.load(Path2 + "ALLdescriptors.npy") # 只有SIFT 特征
features = np.load(Path2 + "ALLfeatures.npy", allow_pickle=True) # 文件路径 + SIFT
print("sucess load descriptors and features")

image_paths = []

for fea in features:
    image_paths.append(fea[0])

print("size of image_path ", len(image_paths))


for num in num_cluster:
    Center = np.load(Path1 + "Center" + str(num) + ".npy") # 聚类中心
    Classnum = np.load(Path1 + "Class" + str(num) + ".npy") # 每个特征的类别
    print(max(Classnum), min(Classnum))

    print("sucess load center and class: ", num)
    print(Classnum.shape)
    i = 0 # feature index
    j = 0 # class index

    im_features = np.zeros((features.shape[0], num), "float32") # 存放张图像的词频率

    print("size of im_features", im_features.shape)

    for fea in features:

        if fea[1] is None:
            i += 1
            continue
        else:
            for k in range(fea[1].shape[0]):
                im_features[i][Classnum[j]] += 1
                j += 1
            i += 1
        if i % 1000 == 0:            
            print(im_features[i - 1])
    
    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((im_features > 0) * 1, axis=0)    
    idf = np.array(np.log((1.0* features.shape[0] +1) / (1.0*nbr_occurences + 1)), 'float32')

    print("finish TF-idf")

    # Perform L2 normalization
    im_features = im_features*idf
    im_features = preprocessing.normalize(im_features, norm='l2')
    print("finish L2 normalization")

    joblib.dump((im_features, image_paths, idf, num, Center), "bof" + str(num) + ".pkl", compress=3)
    print("sucess save ", num)