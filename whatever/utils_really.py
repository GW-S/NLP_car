# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/7
import re

import pandas as pd

def read_train_set():
    '''读取训练集'''
    trainSet_path = '/Users/sheng/PycharmProjects/NLP_car/training_data/train.csv'
    origin  = pd.read_csv(trainSet_path)
    return origin

def read_test_set():
    '''读取训练集'''
    trainSet_path = '/Users/sheng/PycharmProjects/NLP_car/training_data/test_public.csv'
    origin  = pd.read_csv(trainSet_path)
    return origin

def remove_punctuation(line:str):
    """
    移除标点符号
    """
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+|[：]","",line)
    return string




