'''
本文件实现了将json格式标注转化为yolo格式
yolo格式： labelnum, xcenter, ycenter, width, height
注意: 这里的坐标及长度是相对值，范围为[0, 1]
'''
import os
import json
from typing import Text

json_dir = '.\\Dataset\\BJTUdataset\\newdata2.0\\test\\'
out_dir =  '.\\Dataset\\BJTUdataset\\newdata2.0\\test\\labels\\'

label = {'BJTU':0, "TSG":1, "TYHT":2}
FileList = os.listdir(json_dir)

for file in FileList:
    if file.endswith("json"):
        with open(json_dir + file, 'r') as load_f:
            content = json.load(load_f)
        
        textfile = os.path.splitext(file)[0] + '.txt'
        iw = content["imageWidth"]
        ih = content["imageHeight"]

        imginfo = content["shapes"][0]

        l = label[imginfo['label']]

        x1 = min(imginfo['points'][0][0], imginfo['points'][1][0])
        x2 = max(imginfo['points'][0][0], imginfo['points'][1][0])
        y1 = min(imginfo['points'][0][1], imginfo['points'][1][1])
        y2 = max(imginfo['points'][0][1], imginfo['points'][1][1])

        x = (x1 + x2) / 2 / iw
        y = (y1 + y2) / 2 / ih
        w = (x2 - x1) / iw
        h = (y2 - y1) / ih

        fp = open(out_dir + textfile, mode="w", encoding="utf-8")
        file_str = "%d %.6f %.6f %.6f %.6f"%(l, x, y, w, h)
        fp.write(file_str)
        fp.close()