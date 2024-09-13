import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time
import numpy as np
import torch
import sys
from park import *


last_detection_time = 0
detections = []
detections = []
i_det = []
r_location = []
r_num = []
detections_final = []

video_path = r"C:\Users\marsh\Desktop\recognition\yolo\ve1.mp4"
cap = cv2.VideoCapture(video_path)

# cap = cv2.VideoCapture(0) real-time video
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
                    x1 = x1 - 20
                    x2 = x2 + 20
                    y1 = y1 - 20
                    y2 = y2 + 30
                    # 裁剪目标区域
                    cropped_img = frame[y1:y2, x1:x2]

                    scale_factor = 2
                    en_img = cv2.resize(
                        cropped_img,
                        None,
                        fx=scale_factor,
                        fy=scale_factor,
                        interpolation=cv2.INTER_LINEAR,
                    )

                    c = rot_new(en_img)

                    c1 = convert(c)

                    if cv2.imwrite("c1.jpg", c1):
                        print("C1 saved successfully.")

                    else:
                        print("Failed to save image.")

                    # 数字识别
                    model_path_num = r"C:\Users\marsh\Desktop\recognition\yolo\runs\detect\train2\weights\last.pt"
                    model_num_recog = YOLO(model_path_num)
                    results_num = model_num_recog(c1)

                    # 单个图片里检测结果
                    re, text = detect_save(results_num)
                    detections_final.append(re)

    if len(detections_final) >= 1:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        color = (0, 0, 0)
        thickness = 5
        position = (50, 150)  # 文本在图像中的位置
        re, text = detect_save(results_num)

        if isinstance(re, torch.Tensor) and re.dim() == 0:
            re = re.item()

        print(f"Re: {re}, Text: {text}")

        # 在图像上绘制文本
        cv2.putText(
            c1, text, position, font, font_scale, color, thickness, lineType=cv2.LINE_AA
        )

        # 显示图像
        cv2.imshow("Text on Image", c1)

        # 保存图像
        cv2.imwrite("text_image.jpg", c1)
        time.sleep(6)
        break

cap.release()
