import cv2
import numpy as np


def find_common_right_angle_edge_new(img):

    # 预处理：将图像转换为灰度图并进行形态学操作
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯模糊减少噪声
    edges = cv2.Canny(blurred, 20, 160)  # 边缘检测

    # 形态学操作，填充小空洞，减少噪声
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        print(len(approx))
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
        # 显示图像
        cv2.imshow("Detected Contours", img)
        cv2.imwrite("1203.jpg", img)

        # 查找五边形
        if len(approx) == 5:
            cv2.polylines(img, [approx], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.imwrite("five.jpg", img)
            points = [tuple(point[0]) for point in approx]
            print(f"Points: {points}")

            # 计算每条边的向量
            vectors = [np.subtract(points[i], points[(i + 1) % 5]) for i in range(5)]

            # 计算邻边夹角
            angles = [
                np.arccos(
                    np.dot(vectors[i], vectors[(i + 1) % 5])
                    / (
                        np.linalg.norm(vectors[i])
                        * np.linalg.norm(vectors[(i + 1) % 5])
                    )
                )
                for i in range(5)
            ]

            # 弧度转角度
            angles = np.round(np.degrees(angles), decimals=1)
            print(f"Angles: {angles}")

            # 查找近似直角（允许一定范围的误差，比如75到105度）
            right_angles = [i for i, angle in enumerate(angles) if 75 <= angle <= 105]

            if len(right_angles) >= 2:
                for i in range(len(right_angles)):
                    j = (i + 1) % len(right_angles)
                    if (right_angles[j] - right_angles[i]) % 5 == 1:
                        common_edge_idx = right_angles[i]  # 获取第一个直角的边索引
                        common_edge = (
                            points[(common_edge_idx + 1) % 5],
                            points[(common_edge_idx + 2) % 5],
                        )

                        # 在图像上绘制检测到的直角边
                        cv2.line(gray, common_edge[0], common_edge[1], (0, 0, 255), 3)

                        # 保存并显示检测结果
                        cv2.imwrite("Detected_Common_Edge.png", gray)
                        print(f"Common edge found: {common_edge}")
                        return common_edge

    print("No right-angle edge found.")
    return None


def find_common_right_angle_edge(img):
    img = cv2.imread(img)

    edges = cv2.Canny(img, 50, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 5:  # 如果找到了五边形
            points = [tuple(point[0]) for point in approx]
            print(points)
            # 计算每条边的向量
            vectors = [np.subtract(points[i], points[(i + 1) % 5]) for i in range(5)]
            # 计算邻边夹角
            angles = [
                np.arccos(
                    np.dot(vectors[i], vectors[(i + 1) % 5])
                    / (
                        np.linalg.norm(vectors[i])
                        * np.linalg.norm(vectors[(i + 1) % 5])
                    )
                )
                for i in range(5)
            ]

            # 弧度转角度
            angles = np.round(np.degrees(angles), decimals=2)

            # 查找直角（约等于90度）
            right_angles = [i for i, angle in enumerate(angles) if 80 <= angle <= 100]

            if len(right_angles) >= 2:
                for i in range(len(right_angles)):
                    j = (i + 1) % len(right_angles)
                    if (right_angles[j] - right_angles[i]) % 5 == 1:
                        common_edge_idx = right_angles[i]  # 获取第一个直角的边索引
                        common_edge = (
                            points[(common_edge_idx + 1) % 5],
                            points[(common_edge_idx + 2) % 5],
                        )
                print(angles)
                cv2.line(img, common_edge[0], common_edge[1], (0, 0, 255), 2)

                # 显示图像
                cv2.imwrite("Detected Common Edge.png", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                return common_edge
