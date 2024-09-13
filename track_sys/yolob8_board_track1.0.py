# 实时摄像头追踪天井
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import numpy as np

last_detection_time = 0
detections = []
detection_interval = 1

cap = cv2.VideoCapture(0)
model_path = (
    r"C:\Users\marsh\Desktop\recognition\yolo\weight_data\vision3.0_only_board_rough.pt"
)
model = YOLO(model_path)
detections = []


def duplicate(i):
    return i in detections


while cap.isOpened():
    ret, frame = cap.read()

    results = model.track(
        source=0,
        persist=True,
        conf=0.5,
        iou=0.5,
        show=True,
        tracker="botsort.yaml",
    )
    current_time = time.time()
    for r in results:
        i = r.boxes.id

    if not duplicate(i):
        detections.append(i)

    frame_resized = cv2.resize(frame, (640, 480))

    cv2.imshow("YOLOv8 Detection", frame)

    # 'q' exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if len(detections) >= 4:
        print(detections)
        break
