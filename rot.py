import cv2
import numpy as np
from location_based.detect_edge import *

image_path = r"C:\Users\marsh\Desktop\recognition\yolo\image.jpg"
image = cv2.imread(image_path)

# 边缘坐标
start_point, end_point = find_common_right_angle_edge_new(
    r"C:\Users\marsh\Desktop\recognition\yolo\image.jpg"
)


# 计算边缘的角度（与水平线的角度）
def calculate_angle(start_point, end_point):
    delta_x = end_point[0] - start_point[0]
    delta_y = end_point[1] - start_point[1]
    angle = np.arctan2(delta_y, delta_x)  # 弧度
    angle_degrees = np.degrees(angle)  # 转换为角度
    return angle_degrees


angle = calculate_angle(start_point, end_point)

# 计算旋转中心
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# 计算旋转矩阵
rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

# 计算旋转后的图像大小
cos = np.abs(rotation_matrix[0, 0])
sin = np.abs(rotation_matrix[0, 1])
new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

# Adjust rotation matrix to take in account the translation
rotation_matrix[0, 2] += (new_w / 2) - center[0]
rotation_matrix[1, 2] += (new_h / 2) - center[1]

# 旋转图像
rotated_image = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))

# 保存旋转后的图像
cv2.imwrite("rotated_image.jpg", rotated_image)
