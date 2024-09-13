import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import numpy as np
import torch


def detect_save(results):

    frame_detections = []
    re = None

    if results is not None:
        for result in results:
            boxes = result.boxes.xyxy
            confs = result.boxes.conf
            classes = result.boxes.cls

            for box, conf, cls in zip(boxes, confs, classes):
                if conf > 0.7:
                    x1, y1, x2, y2 = map(int, box)

                    detection = {
                        "class": cls,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": conf,
                    }

                    frame_detections.append(detection)
                    if frame_detections:
                        x1_0, x2_0 = 0, 0
                        label1, label2 = None, None
                        for idx, detection in enumerate(frame_detections):
                            # 抽出迭代值
                            if idx == 0:
                                x1_0 = detection["x1"]
                                label1 = detection["digit"]
                            elif idx == 1:
                                x2_0 = detection["x1"]
                                label2 = detection["digit"]

                        if x1_0 > x2_0 and x2_0 != 0:
                            a = label1
                            b = label2
                            re = 10 * b + a
                        elif x1_0 < x2_0 and x2_0 != 0:
                            a = label2
                            b = label1
                            re = 10 * b + a
                        else:
                            re = None
                            a = b = None
                        print(
                            f"The tens place is {b},the unit place is {a},result_final:{re}"
                        )
                    else:
                        print("No digits detected in this frame.")
    return re


model_path_num = r"C:\Users\marsh\Desktop\recognition\yolo\vision1_only_number.pt"
model_num_recog = YOLO(model_path_num)
results_num = model_num_recog(
    r"C:\Users\marsh\Desktop\recognition\yolo\rotated_image.jpg"
)

# 单个帧里检测结果
re = detect_save(results_num)

print(re)
