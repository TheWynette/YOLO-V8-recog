import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import numpy as np
import torch
from garage import *

last_detection_time = 0
detections = []
detections = []
i_det = []
r_location = []
r_num=[]

cap = cv2.VideoCapture(0)
model_path = (
    r"C:\Users\marsh\Desktop\recognition\yolo\weight_data\vision3.0_only_board_rough.pt"
)
model = YOLO(model_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("OMGyouidiot")
        break

    results = model.track(
        source=frame,
        persist=True,
        conf=0.5,
        iou=0.5,
        show=True,
        tracker="botsort.yaml",
    )
    current_time = time.time()

    if time_lim(current_time):
        if results is not None:
            for r in results:
                box = r.boxes
                i = r.boxes.id
                box = box.cpu().numpy()
                if i is not None:
                    i = i.cpu().numpy()
                if not duplicate(i, i_det):
                    detections.append(box)
                    i_det.append(i)

        frame_resized = cv2.resize(frame, (640, 480))

        cv2.imshow("YOLOv8 Detection", frame)

        num_objects = len(box) if box else 0

        if num_objects >= 4:
            for result in results:
                xyxy = result.boxes.xyxyn.cpu().numpy()

            xyxy1 = detections[0]
            xyxy2 = detections[1]
            xyxy3 = detections[2]
            xyxy4 = detections[3]
            print("start recognition or other")
            print("1:", xyxy1)
            print("2:", xyxy2)
            print("3:", xyxy3)
            print("4:", xyxy4)

            rank(xyxy1,xyxy2,xyxy3,xyxy4) #排序 给天井赋位
            
            xyxy_detec= [xyxy1, xyxy2, xyxy3, xyxy4]


            #下面是数字检测

            model_path_num = r"C:\Users\marsh\Desktop\recognition\yolo\best.pt"
            model_num_recog = YOLO(model_path_num)
            ret, frame = cap.read()
            results_num = model_num_recog(frame)

            #单个帧里检测结果
            for result in results_num:
                xyxy1 = result.boxes.xyxy
                r_one=result.boxes.cls
                                                                #结果分为单帧位置和对应数字
                count = -1
                for detect in xyxy_detec:
                    count += 1                                   #与天井逐个比较
                    if compare(xyxy1, detect):                   #数字归属性检验
                        if not duplcate_num(r_one, r_num):       #归属并且不重复的结果放入对应的天井序列行里
                            r[count].append(xyxy1)
                            r_num[count].append(r_one)


            r_ten_co=num_gene(r_location,r_num)

            s_n=find_row(medi(r_ten_co),r_ten_co)

            print(s_n)

            break   #先print还是先break？？？？？？？

        # 'q' exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()