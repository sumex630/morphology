# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/4/6 17:21
@file: process_folder_for_save.py
@brief: 将结果保存为指定的文件夹内
"""
import os
import shutil
import time
from copy import copy
from pprint import pprint


def folder_for_save(input_path, algorithm_type, processing_methods):
    """
    根据输入图片的路径，保存结果图片到指定文件夹内
    :param processing_methods: ['gmm', 'morgholoy', 'median']
    :param input_path:../../dataset/baseline/baseline/{}/input'.format(category)
    :param algorithm_type:gmm
    :return: ../../results/gmm/gmm/baseline/{}/bin%06d.jpg
    """
    output_path_dict = {}
    input_path_list = input_path.replace('\\', '/').split('/')
    output_path_list = input_path_list[:-1]  # 去掉input
    dataset_index = output_path_list.index('dataset')  # 获取 dataset 所在下标
    # output_path = list2path(output_path_list[:dataset_index])  # dataset 前的路径，目的是让results文件夹与dataset文件夹同级
    output_path = list2path(['..', '..'])
    # output_path = ['..', '..']
    # 替换 dataset 为 results
    output_path_list[dataset_index] = 'results'
    # 替换第一个 baseline 为 algorithm_type
    output_path_list.insert(dataset_index + 1, algorithm_type)
    for method in processing_methods:  # 按 方法创建文件，进行分类
        output_path_list[dataset_index + 2] = method
        output_path_ = copy(output_path)

        for p in output_path_list[dataset_index:]:
            output_path_ = os.path.join(output_path_, p)
            create_dir_is_exist(output_path_)

        dir_is_null(output_path_)

        output_path_dict[method] = output_path_

    # pprint(output_path_dict)
    return output_path_dict


def dir_is_null(path):
    """
    文件夹下是否为空，不为空删除文件
    :param path:
    :return:
    """
    flag = os.listdir(path)
    if flag:
        # 强制删除文件夹
        shutil.rmtree(path)
        create_dir_is_exist(path)


def create_dir_is_exist(path):
    """
    判断文件夹是否存在，不存在则创建
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)


def list2path(path_list):
    """
    根据list返回 path
    :param path_list:
    :return:
    """
    return os.path.join(*path_list) if path_list else ''


if __name__ == '__main__':
    PATH_ROOT = '../../../dataset/baseline/baseline/highway/input'
    # start_time = time.time()
    # create_dir_is_exist(PATH_ROOT)

    folder_for_save(PATH_ROOT, 'gmm', ['gmm', 'morgholoy', 'median'])
