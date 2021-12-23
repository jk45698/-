'''
使用Pytorch 实现的kmeans聚类
由于算法没有优化，导致所需资源过大
遂弃用
'''
import torch
import numpy as np 
from kmeans_pytorch import kmeans

DataPath = ["../mission1/SIFT_Feature/descriptors2000.npy", 
            "../mission1/SIFT_Feature/descriptors4000.npy", 
            "../mission1/SIFT_Feature/descriptors6000.npy", 
            "../mission1/SIFT_Feature/descriptors.npy" ]

FeaPath = ["../mission1/SIFT_Feature/features2000.npy",
           "../mission1/SIFT_Feature/features4000.npy",     
           "../mission1/SIFT_Feature/features6000.npy",     
           "../mission1/SIFT_Feature/features.npy"]

print("loading " + DataPath[0])
Data = np.load(DataPath[0])
print("finish " +  DataPath[0] + " " , Data.shape)

for i in range(1, 4):
    print("loading " + DataPath[i])
    Data = np.vstack((Data, np.load(DataPath[i])))
    print("finish " +  DataPath[i] + " " , Data.shape)

# 将 List 转换为 Tensor
Data_tensor = torch.tensor(Data)
print(Data_tensor.type)
print("数据加载完成！")

# 聚类中心大小
num_cluster = [64, 128, 256, 512, 1024, 2048]

for num in num_cluster:

    cluster_ids_x = torch.tensor([])
    cluster_centers = torch.tensor([])
    
    print("Starting cluster, num: ", num)

    cluster_ids_x, cluster_centers = kmeans(
        X = Data_tensor, num_clusters=num,
        distance='euclidean', device=torch.device('cuda:0')
    )

    print("End cluster, num: ", num)

    # 保存聚类中心
    np.save("../mission1/kmeans/Center" + str(num), cluster_centers.numpy())
    print(num, cluster_centers.shape)

    # 保存每个特征的类别
    np.save("../mission1/kmeans/Class" + str(num), cluster_ids_x.numpy())