import glob
import os
import cv2
import json

def main():
	json_dir = 'E:/newdata2.0/test/'
	Path = "./YOLO_files_ground_truth/"

	files = os.listdir(Path)

	for file in files:

		fileName = file

		outFile = './mAP/input/ground-truth/' + fileName

		with open(json_dir + os.path.splitext(file)[0] + ".json", 'r') as load_f:		
			content = json.load(load_f)

		dims = (content["imageWidth"], content["imageHeight"])

		print('Converting file ' + fileName)

		readFile = open(Path + file, 'r')
		writeFile = open(outFile, 'w')

		lines = readFile.read().split('\n')

		for line in lines:

			if len(line) != 0:

				# vector in the YOLOv5 format: class x_center y_center width height
				v = line.split(' ')

				# Inputs in YOLOv5 format
				inClass = v[0]
				inXC = float(v[1])
				inYC = float(v[2])
				inW = float(v[3])
				inH = float(v[4])

				# Outputs in mAP format: <class_name> <left> <top> <right> <bottom> [<difficult>]

				XC = int(inXC*dims[0])
				YC = int(inYC*dims[1])

				semiW = int(inW*dims[0]/2)
				semiH = int(inH*dims[0]/2)

				left = str(XC - semiW)
				top = str(YC - semiH)
				right = str(XC + semiW)
				bottom = str(YC + semiH)

				outLine = inClass + ' ' + left + ' ' + top + ' ' + right + ' ' + bottom + '\n'
				writeFile.write(outLine)

		readFile.close()
		writeFile.close()

if __name__ == '__main__':
	main()