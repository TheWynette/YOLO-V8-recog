# 后端处理

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import numpy as np

a = np.array([1111, 2222, 3333])

model_path = r"C:\Users\marsh\Desktop\recognition\yolo\best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)  #  0 ：first camera


def calculate_iou(x1, y1, x2, y2, x1_e, y1_e, x2_e, y2_e):

    xi1 = max(x1, x1_e)
    yi1 = max(y1, y1_e)
    xi2 = min(x2, x2_e)
    yi2 = min(y2, y2_e)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_e - x1_e) * (y2_e - y1_e)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


def duplicate(detections, new_detection, iou_threshold):
    for existing in detections:
        iou = calculate_iou(
            existing["x1"],
            existing["y1"],
            existing["x2"],
            existing["y2"],
            new_detection["x1"],
            new_detection["y1"],
            new_detection["x2"],
            new_detection["y2"],
        )
        if iou > iou_threshold and existing["digit"] == new_detection["digit"]:
            return True
    return False


def detect_save(frame, current_time):
    global last_detection_time
    if current_time - last_detection_time < detection_interval:
        return last_detection_time, None
        # met ,return none
    last_detection_time = current_time

    frame_detections = []
    re = None

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, classes):
            if conf > 0.7:
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]:{conf:.2f}}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                label = int(f"{model.names[int(cls)]}")

                detection = {
                    "digit": label,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": conf,
                }

                if not duplicate(
                    frame_detections, detection, iou_threshold=0.5
                ):  # problem1~~~
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
    return last_detection_time, re


def duplicate_n(new, detections):
    return new in detections  # 重复则输出1，则not输出0


def save_to_final(detections, new):
    if new is not None and not duplicate_n(new, detections):  # not duplcate则为不重复
        detections.append(new)


last_detection_time = 0
detections = []
detection_interval = 1
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)

    current_time = time.time()

    # 总体数据处理

    last_detection_time, re = detect_save(frame, current_time)

    save_to_final(detections, re)

    frame_resized = cv2.resize(frame, (640, 480))

    cv2.imshow("YOLOv8 Detection", frame)

    # 'q' exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if len(detections) >= 3:
        print(detections)
        median = np.median(detections)
        index = detections.index(median)
        detections.sort(reverse=True)
        print(f"median:{detections[1]},address :{a[index]}")
        break


cap.release()
cv2.destroyAllWindows()
