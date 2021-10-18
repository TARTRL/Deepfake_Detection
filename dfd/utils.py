#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @Contact  : huangsy1314@163.com
# @Website  : https://huangshiyu13.github.io
# @File    : utils.py

import os
import shutil
import xmltodict
from subprocess import check_output


def get_proper_gpu(gpu_number, minimum_memory_per_gpu):
    assert minimum_memory_per_gpu > 0

    xml = check_output('nvidia-smi -x -q', shell=True, timeout=30)
    json = xmltodict.parse(xml)
    gpus = json['nvidia_smi_log']['gpu']
    if not type(gpus) == list:
        gpus = [gpus]

    gpu_indexes = []
    for gpu in gpus:

        minor_number = gpu['minor_number']
        fb_memory_usage = gpu['fb_memory_usage']

        free_memory = int(fb_memory_usage['free'].split(' ')[0])

        if free_memory >= minimum_memory_per_gpu:
            gpu_indexes.append(int(minor_number))

    if gpu_number == -1:
        return gpu_indexes
    else:
        if len(gpu_indexes) < gpu_number:
            print('only {} gpus are available'.format(len(gpu_indexes)))
            return None
        return gpu_indexes[:gpu_number]

def get_base_name(filepath):
    while filepath.endswith('/'):
        filepath = filepath[:-1]
    return os.path.basename(filepath)


def get_all_files(input_dir, suffix=None):
    files = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and (suffix is None or os.path.splitext(file_path)[1] == suffix):
            files.append(file_path)
    return files


def check_file(filename):
    return os.path.isfile(filename)


def check_dir(dirname):
    return os.path.isdir(dirname)


def del_dir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)

def del_file(filename):
    if os.path.isfile(filename):
        os.system('rm ' + filename)

def create_dir(dirname):
    os.mkdir(dirname)


def new_dir(dirname):
    if os.path.isdir(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


def get_filename(filepath):
    filepath = filepath.strip()
    while filepath and filepath[-1] == '/':
        filepath = filepath[:-1]

    file_s = filepath.split('/')
    if '.' in file_s[-1]:
        filename = file_s[-1].split('.')[0]
    else:
        filename = file_s[-1]
    return filename

