# 追踪全损视频
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

cap = r"C:\Users\marsh\Desktop\recognition\yolo\123.mp4"
model_path = (
    r"C:\Users\marsh\Desktop\recognition\yolo\weight_data\vision3.0_only_board_rough.pt"
)
model = YOLO(model_path)

results = model.track(
    source="123.mp4",
    persist=True,
    conf=0.1,
    show=True,
    tracker="botsort.yaml",
)[60]

annotated_frame = results[0].plot()
cv2.imshow("YOLOv8 Tracking", annotated_frame)
for result in results:
    b = result.boxes
    result.save("./result.jpg")
print(b)

cv2.destroyAllWindows()
