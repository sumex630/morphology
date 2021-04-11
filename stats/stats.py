# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/4/6 12:54
@file: stats.py
@brief: 统计: 对算法预测的结果与真实标注结果作对比
"""
import os

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix


def main(dataset_root, results_root):
    """
    :param dataset_root:
    :param results_root:
    :return:
    """
    pass


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
    for filename in file_names[start-1:end]:
        file_path = os.path.join(path, filename)
        # print(file_path)
        img = cv2.imread(file_path, 0)
        retval, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)  # 二值化
        image_set.append(thresh)
    return image_set


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


def get_temporalROI(path):
    """
    获取检测帧的范围
    :param path:dataset/baseline/baseline/highway
    :return:['470', '1700'] [起始帧，结束帧]
    """
    path = os.path.join(path, 'temporalROI.txt')
    with open(path, 'r') as f:
        avail_frames = f.read()

    return avail_frames.split(' ')


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

    stats_dic = {'Recall': round(recall, 2),
                 'Precision': round(precision, 2),
                 'FMeasure': round(fmeasure, 2),
                 'Specificity': round(specficity, 2),
                 'FPR': round(fpr, 2),
                 'FNR': round(fnr, 2),
                 'PWC': round(pbc, 2)}

    return stats_dic


def get_cm_name(video_path_pred):
    """
    根据路径获取 标识 （cat_video）
    :param video_path_pred:
    :return:
    """
    cat_video = video_path_pred.replace('\\', '/').split('/')[-2:]
    return '_'.join(cat_video)


def compare_with_groungtruth(video_path, video_path_pred):
    """
    求混淆矩阵
    :param video_path_pred:
    :param gt_path:
    :return:
    """
    cm_dict = {}

    vaild_frames = get_temporalROI(video_path)  # 有效帧范围
    start_frame_id = int(vaild_frames[0])  # 起始帧号
    end_frame_id = int(vaild_frames[1])  # 结束帧号

    gt_path = os.path.join(video_path, 'groundtruth')  # 基准结果路径
    # 加载gt img
    gt_img = load_img(gt_path, start_frame_id, end_frame_id)

    # for video_path_pred in video_path_preds:
    # 加载results img
    # print("====", video_path_pred)
    res_img = load_img(video_path_pred, start_frame_id, end_frame_id)

    gt_1_dim = np.array(gt_img).reshape(-1)  # y_true
    res_1_dim = np.array(res_img).reshape(-1)  # y_pre
    confusion_mat = confusion_matrix(gt_1_dim, res_1_dim, labels=[255, 0]).ravel()  # 混淆矩阵
    # print("confusion_mat", confusion_mat)

    cm_name = get_cm_name(video_path_pred)
    cm_dict[cm_name] = confusion_mat

    return cm_dict


if __name__ == '__main__':
    DATASETROOT = ''
    RESULTSROOT = ''
    main(DATASETROOT, RESULTSROOT)
