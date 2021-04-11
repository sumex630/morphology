# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/4/8 12:28
@file: util.py
@brief: 
"""
import os
import time
from pprint import pprint

import cv2


def save_img(frame, path):
    """
    保存处理结果
    :param frame: 当前帧
    :param path: 路径
    :return:
    """
    cv2.imwrite(path, frame)


def optimize_morghology(single_frame):
    """
    # 借助形态学处理操作消除噪声点、连通对象
    :param single_frame:
    :return:
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    frame_parsed = cv2.morphologyEx(single_frame, cv2.MORPH_OPEN, kernel, iterations=1)
    frame_parsed = cv2.morphologyEx(frame_parsed, cv2.MORPH_CLOSE, kernel, iterations=3)

    # thresh = cv2.threshold(frame_parsed, 127, 255, cv2.THRESH_BINARY)[1]

    return frame_parsed


def optimize_median(single_frame):
    """
    中值滤波对图形进行优化
    :param single_frame:
    :return:
    """
    blur = cv2.medianBlur(single_frame, 3)  # 中值滤波
    # thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)[1]  # 阈值处理

    return blur


def get_input_path(dataset_rootpath):
    """
    文件操作，获取输入数据的路径
    :param dataset_rootpath:
    :return:
    """
    input_list = []
    for category in get_directories(dataset_rootpath):
        category_path = os.path.join(dataset_rootpath, category, category)

        for video in get_directories(category_path):
            video_path = os.path.join(category_path, video, 'input')

            input_list.append(video_path)

    return input_list


def get_directories(path):
    """
    Return a list of directories name on the specifed path
    :param path:
    :return:
    """
    return [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]


def print_execute_time(func):
    """
    定义一个计算执行时间的函数作装饰器，传入参数为装饰的函数或方法
    :param func:
    :return:
    """
    # 定义嵌套函数，用来打印出装饰的函数的执行时间
    def wrapper(*args, **kwargs):
        # 定义开始时间和结束时间，将func夹在中间执行，取得其返回值
        start = time.time()
        func_return = func(*args, **kwargs)
        end = time.time()
        # 打印方法名称和其执行时间
        print(f'{func.__name__}() execute time: {end - start}s')
        # 返回func的返回值
        return func_return
    # 返回嵌套的内层函数
    return wrapper


def save_execute_time(filename_path, txt):
    """
    保存程序运行时间
    :param filename_path:
    :param txt:
    :return:
    """
    with open(filename_path, 'a') as f:
        f.write(txt + '\n')


@print_execute_time
def cal_sum(size):
    from random import random
    li = [random() for i in range(size)]
    return sum(li)


if __name__ == '__main__':
    # 打印一下1000000个的和，同时会显示执行时间
    # print(cal_sum(1000000))
    save_execute_time("asa12")


