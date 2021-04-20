# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/4/9 15:59
@file: gmm.py
@brief: 高斯混合模型，使用OPenCV库
"""
import os
import time

import cv2

from morphology.arithmetics.common.process_folder_for_save import folder_for_save
from morphology.arithmetics.common.util import save_img, optimize_morghology, optimize_median, get_input_path, \
    save_execute_time


def gmm():
    median_time = 0
    morghology_time = 0
    # 构造高斯混合模型
    model = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # model.setNMixtures(4)  # 设置混合高斯的个数
    # model.setBackgroundRatio(0.75)
    # 只读参数，默认是0.9，高斯背景模型权重和阈值，nmixtures个模型按权重排序后，
    # 只取模型权重累加值大于backgroundRatio的前几个作为背景模型。也就是说如果该值取得非常小，

    file_names = os.listdir(input_path)
    for i, filename in enumerate(file_names):

        file_path = os.path.join(input_path, filename)
        # 以灰度图的形式读取
        frame = cv2.imread(file_path)
        # learningRate 学习速率，值为0-1,为0时背景不更新，为1时逐帧更新，默认为-1，即算法自动更新；
        fgmask = model.apply(frame, learningRate=-1)

        # if flag == algorithm_type:
        fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]

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

        # 保存
        save_img(fgmask, save_path_dict[algorithm_type] + '/%s%06d.jpg' % (algorithm_type, i + 1))
        save_img(morghology_frame, save_path_dict['morghology'] + '/%s%06d.jpg' % ('morghology', i + 1))
        save_img(median_frame, save_path_dict['median'] + '/%s%06d.jpg' % ('median', i + 1))

    # 统计时间入档
    cat_video_l = input_path.split('\\')[-3:-1]  # 类名_视频名_?  [类别，视频名称，方法]
    cat_video = "_".join(cat_video_l)
    text_time = """%s: %f \n%s: %f \n""" %(cat_video + '_morghology', morghology_time, cat_video + '_median', median_time)  # 文件内容
    # print(text_time)
    save_execute_time(stats_time_path, text_time)  # 记录耗时

    cv2.destroyAllWindows()


if __name__ == '__main__':
    ROOTPATH = '../../dataset'
    algorithm_type = 'gmm_lr_def_kel=3_varT=def_iter=3'
    input_path_list = get_input_path(ROOTPATH)
    stats_time_path = '../../results_time/stats_time_{}.txt'.format(algorithm_type)
    processing_methods = [algorithm_type, 'morghology', 'median']
    for input_path in input_path_list:
        # if input_path.split('\\')[-3] == 'dynamicBackground':
        print(input_path)
        # 保存结果的路径
        save_path_dict = folder_for_save(input_path, algorithm_type, processing_methods)
        # 算法入口
        gmm()


