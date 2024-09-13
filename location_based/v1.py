import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import numpy as np
import torch
from garage import *
from park import *

last_detection_time = 0
detections = []
detections = []
i_det = []
r_location = []
r_num = []

detections_final = []

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

    # 获取检测到的目标区域坐标
    if results is not None:
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()  # 获取目标框的坐标
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else None

            # 那个公式要像素坐标还是归一化的？？？？？？？
            if boxes is not None and ids is not None:
                for box, id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)  # 转换为整数像素坐标
                    x_f = (x1 + x2) / 2
                    y_f = (y1 + y2) / 2
                    x1 = x1 - 16
                    x2 = x2 + 16
                    y1 = y1 - 16
                    y2 = y2 + 16
                    # solution = solve_equations(x_f, y_f) 坐标转换问题，先不用
                    # x_a = {solution[0]}
                    # y_a = {solution[1]}

                    # x_a_r, y_a_r = location_compare(x_a, y_a)

                    # 裁剪目标区域
                    cropped_img = frame[y1:y2, x1:x2]
                    c = rot_new(cropped_img)

                    # 放大目标区域
                    scale_factor = 2
                    if c is None:  # 检测是否为空
                        print("Error: Image not loaded correctly.")
                    else:
                        en_img = cv2.resize(
                            c,
                            None,
                            fx=scale_factor,
                            fy=scale_factor,
                            interpolation=cv2.INTER_LINEAR,
                        )

                    # 数字识别
                    model_path_num = r"C:\Users\marsh\Desktop\recognition\yolo\best.pt"
                    model_num_recog = YOLO(model_path_num)
                    results_num = model_num_recog(en_img)

                    # 单个帧里检测结果
                    re = detect_save(results_num)

                    # detection_h = {"x": x_a_r, "y": y_a_r, "result": re} 转换后坐标，先不用
                    detection_h = {"x": x_f, "y": y_f, "result": re}
                    add_dict(detection_h, detections_final)

    if len(detections_final) >= 1:

        s_n = find_mid(detections_final)

        print(s_n["x_a_r"], s_n["y_a_r"])
        break

        # 'q' exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
