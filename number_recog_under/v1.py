import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

import numpy as np
import torch
from garage import *

# from yolov8_tracker1_9_test import *
r_location = []
r_num=[]

cap = cv2.VideoCapture(0)

model_path = r"C:\Users\marsh\Desktop\recognition\yolo\best.pt"
model = YOLO(model_path)
ret, frame = cap.read()
results = model(frame)

#单个帧里检测结果

for result in results:
    xyxy1 = result.boxes.xyxy
    r_one=result.boxes.cls
                                                     #结果分为单帧位置和对应数字
    count = 0
    for detect in xyxy_detec:
        count += 1                                   #与天井逐个比较
        if compare(xyxy1, detect):                   #数字归属性检验
            if not duplcate_num(r_one, r_num):       #归属并且不重复的结果放入对应的天井序列行里
                r[count].append(xyxy1)
                r_num[count].append(r_one)


r_ten_co=num_gene(r_location,r_num)

s_n=find_row(medi(r_ten_co),r_ten_co)

print(s_n)
