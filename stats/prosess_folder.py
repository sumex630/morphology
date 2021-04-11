# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/4/6 12:50
@file: prosess_folder.py
@brief: 对数据集的文件操作，方便后续的算法检测
"""
import os
from pprint import pprint
from time import time

import pandas as pd
from stats import compare_with_groungtruth, get_stats, get_cm_name, get_temporalROI, get_video_info


def process_folder(dataset_root, results_root, arithmetic):
    """
    Call your executable for all sequences in all categories
    :param results_root: 结果根路径
    :param dataset_root: 数据集根目录
    :return:
    """

    # for arithmetic in get_directories(results_root):  # 遍历每种算法的结果
    # 可以向不同的sheet写入数据
    for category in get_directories(dataset_root):  # 遍历每种类型
        category_path = os.path.join(dataset_root, category, category)  # dataset/baseline/baseline
        algorithm_path = os.path.join(results_root, arithmetic)  # results/gmm
        # print(category_path)
        for video in get_directories(category_path):  # 每个类别下的video
            video_path = os.path.join(category_path, video)  # dataset/baseline/baseline/highway

            # 三种处理方法 results/gmm/123/baseline/highway
            reults_video_path_list = []
            for process_method in get_directories(algorithm_path):
                reults_video_path = os.path.join(algorithm_path, process_method, category, video)
                try:
                    flag = os.listdir(reults_video_path)  # 判断该结果文件夹是否为空，若为空则不进行相对应的gt计算
                    if flag:
                        reults_video_path_list.append(os.path.join(algorithm_path, process_method, category, video))
                except Exception as e:
                    continue
            # 计算混淆矩阵
            # 获取统计
            # print(reults_video_path_list)
            for video_path_pred in reults_video_path_list:
                process_method_ = video_path_pred.replace('\\', '/').split('/')[-3]

                print('{}__{}__{} 计算混淆矩阵中...'.format(category, video, process_method_))
                confusion_matrix_dict = compare_with_groungtruth(video_path, video_path_pred)
                # pprint(confusion_matrix_dict)
                cm_name = get_cm_name(video_path_pred)
                # print(process_method_)

                confusion_matrix = confusion_matrix_dict[cm_name]
                stats_dict = get_stats(confusion_matrix)
                # 加入 分辨率，总帧数，起始帧，结束帧，有效帧
                video_info_dict = get_video_info(video_path)

                # 合并
                stats_dict.update(video_info_dict)

                # 写入
                save_path = "../stats_results/{}_{}.csv".format(arithmetic, process_method_)
                df1 = pd.DataFrame(stats_dict, index=[cm_name])
                df1.to_csv(save_path, mode='a', header=False)


def save_stats(path, df1):
    """
    保存数据
    :param data:
    :return:
    """
    # if not os.path.exists(path):
    #     # 创建一个空的excel文件
    #     nan_excle = pd.DataFrame()
    #     nan_excle.to_excel(path)
    #
    # # 打开excel
    # read_df = pd.read_excel(path)
    # stats = read_df.append(df1)
    # sheets是要写入的excel工作簿名称列表
    # df1.to_csv(path, mode='a')
    pass


def get_directories(path):
    """
    Return a list of directories name on the specifed path
    :param path:
    :return:
    """
    return [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]


if __name__ == '__main__':
    DATASETROOT = '../dataset'
    RESULTSROOT = '../results'
    arithmetic = 'gmm'
    process_folder(DATASETROOT, RESULTSROOT, arithmetic)
