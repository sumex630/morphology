# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/3/31 19:06
@file: img2video.py
"""
import os
import cv2
import numpy as np

from arithmetics.common.util import get_directories


def load_data_set(path):
    """
    # 读取图片集合，
    :param path: 图片路径
    :return: 图片集合
    """
    image_set = []
    file_names = os.listdir(path)
    for filename in file_names:
        file_path = os.path.join(path, filename)
        # 以灰度图的形式读取
        img = cv2.imread(file_path, cv2.COLOR_BGR2GRAY)

        if img is not None:
            img = cv2.resize(img, (width, height))
            try:
                # print('当前帧：{}'.format(filename))
                video_write.write(img)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    path = '../dataset/dynamicBackground/dynamicBackground'
    for video in get_directories(path):
        print(video)
        video_path = os.path.join(path, video, 'input')
        first_frame_path = os.path.join(video_path, os.listdir(video_path)[0])
        img = cv2.imread(first_frame_path)
        img_shape = img.shape
        height = img_shape[0]  # 240 * 320
        width = img_shape[1]
        print(img_shape)

        video_write = cv2.VideoWriter('../img2video/input/{}.avi'.format(video),
                                      cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),  # 'P', 'I', 'M', '1'
                                      30, (width, height),
                                      True)  # 240, 320
        # path = 'dataset/highway/optimized'
        load_data_set(video_path)

        video_write.release()
        cv2.destroyAllWindows()


