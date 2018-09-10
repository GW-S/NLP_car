# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/10

import pandas as pd
import re

def remove_punctuation(line: str):
    """
    移除标点符号
    """
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+|[：]", "", line)
    return string

def split_sentence(sentence,gram):
    '''
    切分句子,根据我需求的长度进行切分
    gram 就是我需求的长度
    比如： 我是一个猪
    直接切成，我，是，一，个，猪，这样的东西
    '''
    result = []
    for idx in range(len(sentence)):
        if idx + gram <= len(sentence):
            result.append(sentence[idx:idx+gram] )
    return result

def split_content(train_data,gram):
    """
    切分文档
    """
    result_list = []
    for eachline in train_data['content']:
        eachline = remove_punctuation(eachline)
        result_list.append(split_sentence(eachline,gram))
    return result_list

def read_train_set():
    '''读取训练集'''
    trainSet_path = '/Users/sheng/PycharmProjects/NLP_car/training_data/train.csv'
    origin  = pd.read_csv(trainSet_path)
    return origin

def read_train_set_char():
    """读取单个词集"""
    trainSet_path = '/Users/sheng/PycharmProjects/NLP_car/sklearn_model/char_single.csv'
    origin = pd.read_csv(trainSet_path)
    return origin

def tokenizer_topic(X, topic: list):
    # X.value数据类型为numpy,所以要化为list
    this_topic = X['subject']
    if this_topic in topic:
        X = X.append(pd.Series({"subject_id": topic.index(this_topic)}))  # jieba控制写的列数，需要修改
    else:
        print("wrong in tokenizer_topic")
    return X











