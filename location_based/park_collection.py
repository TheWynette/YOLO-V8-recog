import numpy as np
from scipy.optimize import fsolve
from v1 import *


# 姿态角（弧度）
phi = np.deg2rad(30)  # 方向角
omega = np.deg2rad(15)  # 俯仰角
kappa = np.deg2rad(45)  # 滚转角


a1 = np.cos(phi) * np.cos(kappa) - np.sin(phi) * np.sin(omega) * np.sin(kappa)
a2 = -np.cos(phi) * np.sin(kappa) - np.sin(phi) * np.sin(omega) * np.cos(kappa)
a3 = -np.sin(phi) * np.cos(omega)
b1 = np.cos(omega) * np.sin(kappa)
b2 = np.cos(omega) * np.cos(kappa)
b3 = -np.sin(omega)
c1 = np.sin(phi) * np.cos(kappa) + np.cos(phi) * np.sin(omega) * np.sin(kappa)
c2 = -np.sin(phi) * np.sin(kappa) + np.cos(phi) * np.sin(omega) * np.cos(kappa)
c3 = np.cos(phi) * np.cos(omega)

# 给定的飞机绝对坐标 (X_S, Y_S, Z_S) 和画面内坐标 (x_F, y_F)

x_s = 1
y_s = 1
z_s = 50
z_a = 0
z_f = 50
x_f = 1
y_f = 1


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

solution = fsolve(equations, initial_guess)

print(f"解为: x = {solution[0]}, y = {solution[1]}")




    if len(approx) != 5:
        print(len(approx), flush=True)
        raise ValueError("未检测到五边形轮廓")

    if edge1 is None or edge2 is None:
        raise ValueError("未能找到两个直角的共边")