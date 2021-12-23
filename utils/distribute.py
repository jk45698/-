'''
本程序将待标注的训练集和测试集均分为三个文档
分别由陈晓雄、张宇彤和杨涵淞进行重新标注
最后组成一个全新的数据集
author: 张宇彤
'''
import os.path
import shutil


def copyFiles(sourceDir, targetDir):
	files = os.listdir(sourceDir)
	files.sort()
	# print(files)
	acc=0
	lists=[]

	for i in range(len(files)):
		prei = files[i][:files[i].index('.')]
		houzhui = files[i][files[i].index('.'):]
		# if houzhui=='.json':
		# 	lists.append(files[i])
		# 	print(files[i])
		#print(len(lists))
		if houzhui=='.json':
			for j in range(len(files)):
				prej = files[j][:files[j].index('.')]
				houzhuij=files[j][files[j].index('.'):]
				if prei == prej and houzhuij!='.json':
					shutil.move(sourceDir+r'\\'+files[i], targetDir)
					shutil.move(sourceDir + r'\\' + files[j], targetDir)
					acc+=1
					if acc==341:
						return
	# print(len(files))


copyFiles(r'E:\vision\Dataset\newdata\train', r'E:\vision\Dataset\合作\TRAINzyt')
