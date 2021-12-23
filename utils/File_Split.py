'''
本程序将原本数据集中的交大相关图像从原数据集中分类出来
以便后续的统一格式、重新标注和训练mission2
author: cxx
'''
import os
import shutil

NameList = ["nm-", "sjz-", "sy-", "tsg-", "tyht-"]

old_path1 = "./Dataset/train/"
old_path2 = "./Dataset/test/"
new_path1 = "./Dataset/newdata/train/"
new_path2 = "./Dataset/newdata/test/"
old_path = [old_path1, old_path2]
new_path = [new_path1, new_path2]
sum = 0
JSONnum = 0
IMGnum = 0

for i in range(2):
    
    Filelist = os.listdir(old_path[i])

    for Filename in Filelist:

        name = str.lower(Filename)
        flag = False

        for item in NameList:
            if item in name:
                flag = True
                break
        
        if flag:
            full_path = os.path.join(old_path[i], Filename)
            despath = os.path.join(new_path[i], Filename)
            shutil.move(full_path, despath)

            if "json" in name:
                JSONnum += 1
            else:
                IMGnum += 1
            sum += 1
            if sum % 100 == 0:
                print(sum, " " + full_path + " " + despath)

print("IMGnum:" , IMGnum)
print("JSONnum:", JSONnum)
print("sum:", sum)