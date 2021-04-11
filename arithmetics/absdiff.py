# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/4/10 18:53
@file: absdiff.py
@brief: 
"""
import os
import time
from pprint import pprint

import cv2

from arithmetics.common.process_folder_for_save import folder_for_save
from arithmetics.common.util import save_img, optimize_morghology, optimize_median, get_input_path, save_execute_time


def absdiff():
    median_time = 0
    morghology_time = 0
    file_names = os.listdir(input_path)

    # 读取第一帧
    first_frame_path =  os.path.join(input_path, file_names[0])
    first_frame = cv2.imread(first_frame_path)
    pre_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    # 高斯滤波，减少噪声干扰
    # pre_gray = cv2.GaussianBlur(pre_gray, ksize=(0, 0), sigmaX=15)

    for i, filename in enumerate(file_names):
        file_path = os.path.join(input_path, filename)
        # 以灰度图的形式读取
        frame = cv2.imread(file_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 高斯滤波，减少噪声干扰
        # gray = cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=15)

        # 计算当前帧与前一帧的差值的绝对值
        fgmask = cv2.absdiff(gray, pre_gray)
        fgmask = cv2.threshold(fgmask, 30, 255, cv2.THRESH_BINARY)[1]

        # if flag == 'median':
        start = time.perf_counter()
        median_frame = optimize_median(fgmask)  # 中值滤波处理
        end = time.perf_counter()
        median_time += (end - start)  # 统计耗时

        # if flag == 'morghology':
        start = time.perf_counter()
        morghology_frame = optimize_morghology(fgmask)  # 形态学处理
        end = time.perf_counter()
        morghology_time += (end - start)  # 统计耗时

        # cv2.imshow('fgmask_thresh', fgmask_thresh)
        # cv2.imshow('median_frame', median_frame)
        # cv2.imshow('morghology_frame', morghology_frame)

        # 保存
        save_img(fgmask, save_path_dict[algorithm_type] + '/%s%06d.jpg' % (algorithm_type, i + 1))
        save_img(morghology_frame, save_path_dict['morghology'] + '/%s%06d.jpg' % ('morghology', i + 1))
        save_img(median_frame, save_path_dict['median'] + '/%s%06d.jpg' % ('median', i + 1))

    # # 统计时间入档
    cat_video_l = input_path.split('\\')[-3:-1]  # 类名_视频名_?  # [类别，视频名称，方法]
    cat_video = "_".join(cat_video_l)
    text_time = """%s: %.3f \n%s: %.3f \n""" %(cat_video + '_morghology', morghology_time, cat_video + '_median', median_time)  # 文件内容
    # print(text_time)
    save_execute_time(stats_time_path, text_time)  # 记录耗时

    cv2.destroyAllWindows()


if __name__ == '__main__':
    ROOTPATH = '../dataset'
    stats_time_path = '../time/stats_time_absdiff.txt'
    input_path_list = get_input_path(ROOTPATH)
    algorithm_type = 'absdiff'
    processing_methods = [algorithm_type, 'morghology', 'median']
    for input_path in input_path_list:
        print(input_path)
        # 保存结果的路径
        save_path_dict = folder_for_save(input_path, algorithm_type, processing_methods)
        absdiff()  # 算法入口



