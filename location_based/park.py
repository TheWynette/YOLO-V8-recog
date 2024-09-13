import numpy as np
import cv2
from scipy.optimize import fsolve
from detect_edge import *


def solve_equations(x_f, y_f):

    def equations(vars):
        x_a, y_a = vars

        eq1 = (
            z_f
            * (a1 * (x_a - x_s) + b1 * (y_a - y_s) + c1 * (z_a - z_s))
            / (a3 * (x_a - x_s) + b3 * (y_a - y_s) + c3 * (z_a - z_s))
            - x_f
        )
        eq2 = (
            z_f
            * (a2 * (x_a - x_s) + b2 * (y_a - y_s) + c2 * (z_a - z_s))
            / (a3 * (x_a - x_s) + b3 * (y_a - y_s) + c3 * (z_a - z_s))
            - y_f
        )

        return [eq1, eq2]

    # 初始猜测值
    initial_guess = [x_s + 1, y_s + 1]

    # 使用fsolve求解非线性方程组
    solution = fsolve(equations, initial_guess)

    return solution


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
                                label1 = detection["class"]
                            elif idx == 1:
                                x2_0 = detection["x1"]
                                label2 = detection["class"]

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
                        text = f"result: {re}"

                    else:
                        print("No digits detected in this frame.")
    return re, text


def add_dict(dict, group):

    dict_f = tuple(list(dict.items())[:2])

    for e in group:
        e_f = tuple(list(e.items())[:2])
        if dict_f == e_f:
            return
    group.append(dict)


def location_compare(x1, y1):

    # compare and output

    return x1, y1


def find_mid(dict):

    s_d = sorted(dict, key=lambda d: d["c"])

    mid_index = len(s_d) // 2
    median_dict = s_d[mid_index]
    return median_dict


def angle_between_edges(edge1, edge2):
    def normalize(vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm != 0 else vec

    vec1 = np.array(edge1[1]) - np.array(edge1[0])
    vec2 = np.array(edge2[1]) - np.array(edge2[0])

    vec1_normalized = normalize(vec1)
    vec2_normalized = normalize(vec2)

    dot_product = np.dot(vec1_normalized, vec2_normalized)
    angle = np.arccos(dot_product) * 180 / np.pi
    return angle


def find_right_angle_edges(pts):

    edges = []
    for i in range(len(pts)):
        p1, p2 = pts[i], pts[(i + 1) % len(pts)]
        edges.append((p1, p2))

    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            angle = angle_between_edges(edges[i], edges[j])
            if np.isclose(angle, 90, atol=3):
                return edges[i], edges[j]

    return None, None


def rot_new(img):

    # 边缘坐标
    start_point, end_point = find_common_right_angle_edge_new(img)

    # 计算边缘的角度（与水平线的角度）
    def calculate_angle(start_point, end_point):
        delta_x = end_point[0] - start_point[0]
        delta_y = end_point[1] - start_point[1]
        angle = np.arctan2(delta_y, delta_x)  # 弧度
        angle_degrees = np.degrees(angle)  # 转换为角度
        return angle_degrees

    angle = calculate_angle(start_point, end_point)

    # 计算旋转中心
    (h, w) = img.shape[:2]
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
    rotated_image = cv2.warpAffine(img, rotation_matrix, (new_w, new_h))

    # 保存旋转后的图像
    cv2.imwrite("rotated_image.jpg", rotated_image)
    return rotated_image


def rot(img):

    image = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray Image", gray)
    cv2.imwrite("image.jpg", gray)

    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有轮廓并找到五边形
    for contour in contours:
        # 近似轮廓为多边形
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 检查是否为五边形
        if len(approx) == 5:
            # 绘制多边形
            cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
            cv2.imshow("Detected Pentagon", img)
            cv2.imwrite("Detected P", img)
            cv2.waitKey(10)

            return approx  # 返回找到的五边形顶点

    if approx is None:
        raise KeyError
    # 找到两个直角边
    edge1, edge2 = find_right_angle_edges(approx.reshape(-1, 2))

    # 计算旋转角度
    def angle_between_points(p1, p2):
        delta_x = p2[0] - p1[0]
        delta_y = p2[1] - p1[1]
        return np.arctan2(delta_y, delta_x) * 180 / np.pi

    angle1 = angle_between_points(edge1[0], edge1[1])
    angle2 = angle_between_points(edge2[0], edge2[1])
    angle = (angle1 + angle2) / 2  # 计算平均角度

    # 计算旋转矩阵
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    # 旋转图像
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
    )

    return rotated_image


def convert(image):
    if image.dtype == np.float32:
        # 将浮点图像转换为0-255的8位无符号整数
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
    elif image.dtype == np.uint16:
        # 将16位图像转换为8位
        image = (image / 256).astype(np.uint8)
    return image
