'''
使用MiniBatchKMeans 加速的kmeans
获取不同 k 的聚类中心和特征类别
'''
import numpy as np 
from sklearn.cluster import MiniBatchKMeans

DataPath = ["../mission1/SIFT_Feature/descriptors2000.npy", 
            "../mission1/SIFT_Feature/descriptors4000.npy", 
            "../mission1/SIFT_Feature/descriptors6000.npy", 
            "../mission1/SIFT_Feature/descriptors.npy" ]

print("loading " + DataPath[0])
Data = np.load(DataPath[0])
print("finish " +  DataPath[0] + " " , Data.shape)

for i in range(1, 4):
    print("loading " + DataPath[i])
    Data = np.vstack((Data, np.load(DataPath[i])))
    print("finish " +  DataPath[i] + " " , Data.shape)

print("数据加载完成！")

# 聚类中心大小
num_cluster = [64, 128, 256, 512, 1024, 2048]

for num in num_cluster:
    print("start ", num)
    model = MiniBatchKMeans(num, batch_size=1000)
    model.fit(Data)
    print("finish ", num)

    # 保存聚类中心
    np.save("../mission1/kmeans/Center" + str(num), model.cluster_centers_)
    print(num, model.cluster_centers_.shape)

    # 保存每个特征的类别
    label = model.predict(Data)
    np.save("../mission1/kmeans/Class" + str(num), label)