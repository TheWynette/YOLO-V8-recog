# 全损视频追踪天井 记录输出数据并停止
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import numpy as np
import torch

last_detection_time = 0
detections = []
detection_interval = 1

cap = cv2.VideoCapture(0)
model_path = (
    r"C:\Users\marsh\Desktop\recognition\yolo\weight_data\vision3.0_only_board_rough.pt"
)
model = YOLO(model_path)

detections = []


def duplicate(i, detections):
    if i is None:
        return True
    else:
        for d in detections:
            if np.array_equal(d, i):
                return True
    return False


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

    if results is not None:
        for r in results:
            box = r.boxes
            box = box.cpu().numpy()
            if not duplicate(box, detections):
                detections.append(box)

    frame_resized = cv2.resize(frame, (640, 480))

    cv2.imshow("YOLOv8 Detection", frame)

    num_objects = len(box) if box else 0

    if num_objects >= 2:
        print("start recognition or other")
        print(detections)
        break

    # 'q' exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
