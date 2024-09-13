# 全损视频追踪天井 记录输出数据并停止
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import numpy as np
import torch
from garage import *

last_detection_time = 0
detections = []

cap = cv2.VideoCapture(0)
model_path = (
    r"C:\Users\marsh\Desktop\recognition\yolo\weight_data\vision3.0_only_board_rough.pt"
)
model = YOLO(model_path)

detections = []
i_det = []

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

            rank(xyxy1, xyxy2, xyxy3, xyxy4)  # 排序 给天井赋位

            xyxy_detec = [xyxy1, xyxy2, xyxy3, xyxy4]
            break

        # 'q' exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
