# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/5/9 12:29
@file: gmm.py
@brief: 
"""
import os
import time

import cv2

from morphology.arithmetics.common.folder_process import list2path, path_is_exist, get_input_path
from morphology.arithmetics.common.stats import stats
from morphology.arithmetics.common.util import optimize_median, optimize_morghology, saveimg, save_fps


def stats_time(input_path, results_rootpath, t):
    """
    :param input_path:
    :param results_rootpath:
    :param t: 耗时
    :return:
    """
    stats_time_path_list = results_rootpath.replace('\\', '/').split('/')
    # 时间统计结果根目录名称
    stats_time_path_list[-2] = 'results_time'
    # 时间统计结果根目录
    stats_time_dir_path = list2path(stats_time_path_list[:-1])
    path_is_exist(stats_time_dir_path)
    # 时间统计结果路径
    stats_time_path = os.path.join(stats_time_dir_path, stats_time_path_list[-1] + '.csv')
    # 统计形态学处理耗时 FPS
    video_path_list = input_path.replace('\\', '/').split('/')
    # category_video 格式命名，FPS
    category_video_name = '_'.join(video_path_list[-3:-1])
    # 获取视频序列中的帧数量，为了后续计算 FPS
    frame_num = len(os.listdir(input_path))
    fps = frame_num / t
    save_fps(stats_time_path, category_video_name, fps)


def gmm(input_path, results_rootpath):
    morghology_time = 0
    # 构造高斯混合模型
    model = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    # model.setNMixtures(4)  # 设置混合高斯的个数
    # model.setBackgroundRatio(0.75)
    # 只读参数，默认是0.9，高斯背景模型权重和阈值，nmixtures个模型按权重排序后，
    # 只取模型权重累加值大于backgroundRatio的前几个作为背景模型。也就是说如果该值取得非常小，

    filenames = os.listdir(input_path)
    for i, filename in enumerate(filenames):
        file_path = os.path.join(input_path, filename)
        frame = cv2.imread(file_path)
        # learningRate 学习速率，值为0-1,为0时背景不更新，为1时逐帧更新，默认为-1，即算法自动更新；
        fgmask = model.apply(frame)
        fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]  # 二值化

        start = time.perf_counter()
        morghology_frame = optimize_morghology(fgmask, 3)  # 形态学处理
        end = time.perf_counter()
        morghology_time += (end - start)  # 统计耗时

        output_filename = 'bin%06d.jpg' % (i + 1)
        saveimg(fgmask, output_filename, input_path, results_rootpath)  # 保存原始检测结果
        saveimg(morghology_frame, output_filename, input_path, results_rootpath, 'morghology')  # 保存形态学处理后的结果

    # fps 入档
    stats_time(input_path, results_rootpath, morghology_time)


def main():
    """
    入口函数
    :return:
    """
    # 数据集根目录
    # F:/Dataset/CDNet2012/baseline\baseline\highway\input
    dataset_rootpath = 'F:/Dataset/CDNet2012/'
    # 检测结果根目录
    results_rootpath = '../../results/gmm_default'
    # 评估指标根目录
    results_stats_rootpath = '../../results_stats'

    # 数据集中视频序列路径列表   /../../input
    input_path_list = get_input_path(dataset_rootpath)
    for input_path in input_path_list:
        print('正在检测：', input_path)
        gmm(input_path, results_rootpath)  # 算法入口

    # 计算评价指标
    stats(dataset_rootpath, results_rootpath, results_stats_rootpath)


if __name__ == '__main__':
    main()