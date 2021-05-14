# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/5/9 13:37
@file: stats.py
@brief: 
"""
import csv
import os
import time
from pprint import pprint
import multiprocessing as mp

import cv2
import numpy as np
import pandas as pd


def get_temporalROI(roi_dir_path):
    """
    获取检测帧的范围
    :param path:dataset/baseline/baseline/highway/groundtruth
    :return:['470', '1700'] [起始帧，结束帧]
    """
    roi_path = os.path.join(roi_dir_path, 'temporalROI.txt')
    with open(roi_path, 'r') as f:
        avail_frames = f.read()

    return avail_frames.split(' ')


def readimg(path, filename):
    """
    :param path:
    :param filename:
    :return:
    """
    file_path = os.path.join(path, filename)
    img = cv2.imread(file_path, 0)
    retval, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)  # 二值化

    return thresh


def load_img(path, start, end):
    """
    加载图片
    :param start: 起始帧
    :param end: 结束帧
    :param path: dataset/baseline/baseline/highway/groundtruth
                dataset/baseline/results/highway
    :return:有效图片的集合
    """
    image_set = []
    file_names = os.listdir(path)
    pool = mp.Pool(int(mp.cpu_count()))
    for filename in file_names[start-1:end]:
        # 并行计算
        thresh = pool.apply_async(readimg, (path, filename)).get()
        image_set.append(thresh)

    pool.close()
    pool.join()

    return image_set


def compute_cm(y_ture, y_pred):
    """
    计算每帧的混淆矩阵
    :param y_ture:
    :param y_pred:
    :return:
    """
    ture_and_pred = cv2.bitwise_and(y_ture, y_pred)  # 与 -- TP（1 的个数）
    ture_or_pred = cv2.bitwise_or(y_ture, y_pred)  # 或 -- TN （0 的个数）
    ture_not = cv2.bitwise_not(y_ture)  # 对 y_ture 取非
    ture_not_and_pred = cv2.bitwise_and(ture_not, y_pred)  # FP （1 的个数）
    ture_not_or_pred = cv2.bitwise_or(ture_not, y_pred)  # FN （0 的个数）

    tp = (ture_and_pred.reshape(-1) == 255).sum()
    fn = (ture_not_or_pred.reshape(-1) == 0).sum()
    fp = (ture_not_and_pred.reshape(-1) == 255).sum()
    tn = (ture_or_pred.reshape(-1) == 0).sum()

    return [tp, fn, fp, tn]


def compare_with_groundtruth(y_true_path, y_pred_path):
    """
    检测结果与groundtruth对比
    :param y_true_path:
    :param y_pred_path:
    :return: 混淆矩阵
    """
    cm = np.array([0, 0, 0, 0])  # 初始化混淆矩阵 [tp, fn, fp, tn]
    roi_dir_path = y_true_path.replace('\\', '/').replace('/groundtruth', '')
    vaild_frames = get_temporalROI(roi_dir_path)  # 评估帧范围
    start_frame_id = int(vaild_frames[0])  # 起始帧号
    end_frame_id = int(vaild_frames[1])  # 结束帧号
    # print("start_frame_id", start_frame_id, end_frame_id)

    y_true_set = load_img(y_true_path, start_frame_id, end_frame_id)
    y_pred_set = load_img(y_pred_path, start_frame_id, end_frame_id)

    # for i, y_true_frame in enumerate(y_true_set):
    #     cm_frame = compute_cm(y_true_frame, y_pred_set[i])  # 每帧的混淆矩阵
    #     cm += cm_frame

    pool = mp.Pool(int(mp.cpu_count()))
    for i, y_true_frame in enumerate(y_true_set):
        # 并行计算
        cm_frame = pool.apply_async(compute_cm, (y_true_frame, y_pred_set[i])).get()
        cm += cm_frame

    pool.close()
    pool.join()

    return cm


def get_stats(cm):
    """
    Return the usual stats for a confusion matrix
    • TP (True Positive)：表示正确分类为前景的像素个数。
    • TN (True Negative)：表示正确分类为背景的像素个数。
    • FP (False Positive)：表示错误分类为前景的像素个数。
    • FN (False Negative)：表示错误分类为背景的像素个数。
    1. Recall 即召回率，表示算法正确检测出的前景像素个数占基准结果图像中所有前景像素个数的百分比，数值在 0 到 1 之间，结果越趋近于 1 则说明算法检测效果越好
    2. Precision 即准确率，表示算法正确检测出的前景像素个数占所有检测出的前景像素个数的百分比，数值在 0 到 1 之间，结果越趋近于 1 则说明算法检测效果越好
    3. F-Measure (F1-Score) 就是这样一个指标，常用来衡量二分类模型精确度，它同时兼顾了分类模型的 Recall 和 Precision，是两个指标的一种加权平均，
        最小值是 0，最大值是 1，越趋近于 1 则说明算法效果越好
    4. Specificity 表示算法正确检测出的背景像素个数占基准结果图像中所有背景像素个数的百分比，数值在 0 到 1 之间，越趋近于 1 则说明算法检测效果越好
    5. FPR 表示背景像素被错误标记为前景的比例，数值在 0 到 1 之间，和上述四个指标相反，该值越趋近于 0，则说明算法检测效果越好
    6. FNR 表示前景像素被错误标记为背景的比例，数值在 0 到 1 之间，同样该值越趋近于 0，则说明算法检测效果越好
    7. PWC 表示错误率，包括前景像素和背景像素，数值在 0 到 ？ 之间，该值越趋近于 0，则说明算法检测效果越好
    :param cm: 混淆矩阵
    :return:
    """
    # TP = cm[0, 0]
    # FN = cm[0, 1]
    # FP = cm[1, 0]
    # TN = cm[1, 1]
    TP, FN, FP, TN = cm

    recall = TP / (TP + FN)
    specficity = TN / (TN + FP)
    fpr = FP / (FP + TN)
    fnr = FN / (TP + FN)
    pbc = 100.0 * (FN + FP) / (TP + FP + FN + TN)
    precision = TP / (TP + FP)
    fmeasure = 2.0 * (recall * precision) / (recall + precision)

    stats_dic = {'Recall': round(recall, 4),
                 'Precision': round(precision, 4),
                 'Specificity': round(specficity, 4),
                 'FPR': round(fpr, 4),
                 'FNR': round(fnr, 4),
                 'PWC': round(pbc, 4),
                 'FMeasure': round(fmeasure, 4),
                 }

    return stats_dic


def get_video_info(path):
    """
    获取视频信息：分辨率，总帧数，起始帧，结束帧，有效帧
    :param path:
    :return:
    """
    video_info_dict = {}

    start_frame = get_temporalROI(path)[0]  # 有效起始帧
    end_frame = get_temporalROI(path)[1]  # 有效终止帧
    video_path = os.path.join(path, 'input')  # video路径
    file_names = os.listdir(video_path)  # video 中帧list
    frame_path = os.path.join(video_path, file_names[0])  # 第一帧
    img = cv2.imread(frame_path)
    img_shape = img.shape

    resolution = '{} * {}'.format(img_shape[0], img_shape[1])  # 分辨率
    total_frames = len(file_names)  # 总帧数
    valid_frame = (int(end_frame) - int(start_frame)) + 1  # 有效帧

    video_info_dict['resolution'] = resolution
    video_info_dict['total_frames'] = total_frames
    video_info_dict['valid_frame'] = valid_frame
    video_info_dict['start_frame'] = start_frame
    video_info_dict['end_frame'] = end_frame

    return video_info_dict


def get_category_video(stats_index):
    """
    csv 索引
    :param stats_index:
    :return: dict
    """
    return {'category_video': stats_index}


def csv_write(stats_root, filename, data):
    """
    保存数据，依次追加
    :param stats_root:
    :param filename:
    :param data:
    :return:
    """
    if not os.path.exists(stats_root):
        os.mkdir(stats_root)
    stats_path = os.path.join(stats_root, filename + '.csv')
    is_stats = True
    try:
        pd.read_csv(stats_path)
    except Exception as e:
        is_stats = False

    header = list(data.keys())
    with open(stats_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        if not is_stats:  # 第一次写入将创建 header
            writer.writeheader()  # 写入列名
        writer.writerows([data])  # 写入数据


def stats(dataset_root, results_root, stats_root):
    # if sub_results_root:
    #     results_path = os.path.join(results_root, sub_results_root)
    # else:
    #     # 批量统计
    #     results_path = results_root

    for dirpath, dirnames, filenames in os.walk(results_root):
        if filenames:
            # print('filenames')  # 包含文件名称[列表形式]
            print('正在计算评估指标：', dirpath)
            dirpath_list = dirpath.replace('\\', '/').split('/')  # 切割路径
            # algorithm_name_index = dirpath_list.index(sub_results_root)  # 获取文件名下标
            algorithm_name_index = dirpath_list.index(os.path.basename(results_root))
            algorithm_name_type = dirpath_list[algorithm_name_index:-2]  #
            save_filename = '_'.join(algorithm_name_type)  # 保存stats时的文件名
            stats_index = '_'.join(dirpath_list[-2:])  # 保存时的索引名称

            dataset_video_path = os.path.join(dataset_root, dirpath_list[-2], dirpath_list[-2], dirpath_list[-1])  # 数据集的视频序列路径
            gt_path = os.path.join(dataset_video_path, 'groundtruth')  # 基准结果路径
            try:
                confusion_matrix = compare_with_groundtruth(gt_path, dirpath)  # 计算混淆矩阵
            except Exception as e:
                print(e)
                continue
            # confusion_matrix = compare_with_groundtruth(gt_path, dirpath)  # 计算混淆矩阵
            frames_stats = get_stats(confusion_matrix)  # 7中度量
            frames_info = get_video_info(dataset_video_path)  # 帧的相关信息
            frames_category = get_category_video(stats_index)  # 索引名
            frames_stats.update(frames_info)  # 合并数据
            frames_category.update(frames_stats)  # 合并数据

            # 保存
            csv_write(stats_root, save_filename, frames_category)


if __name__ == '__main__':
    dataset_root = 'F:/Dataset/CDNet2012/'  # 数据集根目录
    results_root = '../../results/mb_t=t0.3'  # 检测结果根目录
    stats_root = '../../results_stats'  # 统计结果根目录
    # sub_results_root = 'mb'  # 要执行results_root文件夹下的哪个子文件夹, 为空时执行全部

    stats(dataset_root, results_root, stats_root)
