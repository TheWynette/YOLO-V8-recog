# 有十位数检测数字，但无法帧内查重或者设置检测帧间隔

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

model_path = r"C:\Users\marsh\Desktop\recognition\yolo\best.pt"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)  #  0 ：first camera

while cap.isOpened():

    detection_interval = 3
    current_time = time.time()
    global last_detection_time

    def detect_save(frame, current_time):
        if current_time - last_detection_time < detection_interval:
            return

    # met ,return none
    last_detection_time = current_time

    ret, frame = cap.read()
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls
        detections = []
        for box, conf, cls in zip(boxes, confs, classes):
            if conf > 0.7:
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]}"

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
                label = int(label)
                detections.append(
                    {
                        "digit": label,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": conf,
                    }
                )

                if detections:
                    x1_0 = 0
                    x2_0 = 0
                    label1 = None
                    label2 = None
                    for idx, detection in enumerate(detections):

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
                    print(x1_0, label1, x2_0, label2)
                else:
                    print("No digits detected in this frame.")

    frame_resized = cv2.resize(frame, (640, 480))

    cv2.imshow("YOLOv5 Detection", frame)

    # 'q' exit
    if cv2.waitKey(333) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
