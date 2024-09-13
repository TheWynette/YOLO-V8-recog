import numpy as np


def compare(xyxy1, xyxy0):
    x1, y1, x2, y2 = map(int, xyxy1)
    x, y, x0, y0 = map(int, xyxy0)
    if x1 <= x and y1 <= y and x2 <= x0 and y2 <= y0:
        return True
    else:
        return False


def duplicate(i, detections):
    if i is None:
        return True
    else:
        for d in detections:
            if np.array_equal(d, i):
                return True
        return False


def time_lim(current_time):
    detection_interval = 1
    global last_detection_time
    if current_time - last_detection_time < detection_interval:
        return False
    else:
        last_detection_time = current_time
        return True


def duplicate_num(num, detections):
    # 数字不重复性检测

    return False


def rank(w1, w2, w3, w4):

    # 通过空间排序

    return w1, w2, w3, w3


def find_row(number, array):
    indices = np.where(array.flatten() == number)[0]
    return indices

def num_gene(r1,r2)

    #使用位置数组和数据数组构建一个十位数数组
    #需要考虑不同列（天井的箭头指向），也就是它们数字相对排列差异
    if s==1:


    elif s == 2:


    elif s == 3:


    elif s == 4:


    else:
        return False


    return array_r
