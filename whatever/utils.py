# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/6

'''
主要是统计，对于每一个情感词，它单独有的东西，其他的是不是也有。
'''

import collections
from utils.utils_really import read_train_set

from whatever.utils_really import remove_punctuation


def split_sentence(sentence,gram):
    '''切分句子,根据我需求的长度进行切分
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
    result_list = []
    for eachline in train_data['content']:
        eachline = remove_punctuation(eachline)
        result_list.append(split_sentence(eachline,gram))
    return result_list


def counter_one_category(train_data_0,gram):
    Counter_0 = collections.Counter()
    for eachline in split_content(train_data_0, gram):
        Counter_0.update(eachline)
    return Counter_0

def counter_get(counter_negative,negative_set):
    counter_result = {}
    for key, val in counter_negative.items():
        if key in negative_set:
            counter_result[key] = val
    return counter_result




def sort_by_value(d):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    backitems.sort()
    return [ backitems[i][1] for i in range(0,len(backitems))]

def each_only(gram):

    train_data = read_train_set()

    train_data_negative = train_data[train_data['sentiment_value'] == -1]
    train_data_0 = train_data[train_data['sentiment_value'] == 0]
    train_data_positive = train_data[train_data['sentiment_value'] == 1]

    counter_negative = counter_one_category(train_data_negative,gram)
    counter_0 = counter_one_category(train_data_0,gram)
    counter_positive = counter_one_category(train_data_positive,gram)

    negative_set = set(counter_negative)-set(counter_positive)-set(counter_0)           # 每一个独有的数据
    positive_set = set(counter_positive)-set(counter_0)-set(counter_negative)
    set_0 = (set(counter_0)-set(counter_positive)-set(counter_negative))

    counter_result_negative = counter_get(counter_negative,negative_set)
    counter_result_positive = counter_get(counter_positive,positive_set)
    counter_0 = counter_get(counter_0,set_0)

    counter_result_negative = collections.Counter(counter_result_negative ).most_common(100)
    counter_result_positive = collections.Counter(counter_result_positive).most_common(100)
    counter_0 = collections.Counter(counter_0).most_common(100)


    return counter_result_negative,counter_result_positive,counter_0


if __name__ == '__main__':

    # 寻找一下有的类有，而有的类没有的独特词，2代表分词的结构
    print(each_only(2)[1])
    print(each_only(3)[1])
    print(each_only(5)[1])






















    #print(split_content(train_data_0,1)[0])



























    #for each in split_sentence()







