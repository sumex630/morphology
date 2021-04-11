# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/4/10 10:03
@file: test.py
@brief: 
"""
import os


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


if __name__ == '__main__':
    ROOTPATH = '../dataset'
    # get_input_path(ROOTPATH)
