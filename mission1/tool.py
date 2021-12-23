'''
由于设备问题，SIFT提取的特征没有存到同一个文件中
本程序将SIFT特征整合到一个文件中
'''
import numpy as np 
from sklearn.cluster import MiniBatchKMeans

DataPath = ["../mission1/SIFT_Feature/descriptors2000.npy", 
            "../mission1/SIFT_Feature/descriptors4000.npy", 
            "../mission1/SIFT_Feature/descriptors6000.npy", 
            "../mission1/SIFT_Feature/descriptors.npy" ]

FeaPath = ["../mission1/SIFT_Feature/features2000.npy",
           "../mission1/SIFT_Feature/features4000.npy",     
           "../mission1/SIFT_Feature/features6000.npy",     
           "../mission1/SIFT_Feature/features.npy"]

print("loading " + DataPath[0])
Data = np.load(DataPath[0], )
print("finish " +  DataPath[0] + " " , Data.shape)

for i in range(1, 4):
    print("loading " + DataPath[i])
    Data = np.vstack((Data, np.load(DataPath[i])))
    print("finish " +  DataPath[i] + " " , Data.shape)

np.save("../mission1/SIFT_Feature/ALLdescriptors.npy", Data)

Data = []

Data = np.load(FeaPath[0], allow_pickle=True)
for i in range(1, 4):
    print("loading " + FeaPath[i])
    Data = np.vstack((Data, np.load(FeaPath[i], allow_pickle=True)))
    print("finish " +  FeaPath[i] + " " , Data.shape)

np.save("../mission1/SIFT_Feature/ALLfeatures.npy", Data)
