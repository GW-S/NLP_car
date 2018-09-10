# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/9
"""
对原始文件执行统计
统计：
    1.统计各主题的比例
    2.统计不同情感词占据的比例
"""
from whatever.utils_really import read_train_set
from collections import Counter

# 1.统计不同的主题，最大的为动力，2732，最小的为空间，442
train_set = read_train_set()
print(train_set['subject'].value_counts())

# 2.统计不同的情感，最大的为0，6661，1和-1几乎相同，为1616
print(train_set['sentiment_value'].value_counts())

# 3.不同的装备的padding是多少


# 4.统计同一个标题，多情感词的数量
print(train_set['content_id'].value_counts().values.tolist())
now = Counter(train_set['content_id'].value_counts().values.tolist())
print(now)      # {1: 7036, 2: 950, 3: 231, 4: 52, 5: 17, 6: 3, 7: 1}   7:1 是这样的，那就是 0.56 * 0.8 = 0.48

# 5.统计同一个标题，情感词不同的例子,没统计出来，不想统计了
# train_set_group = train_set.groupby(['content_id','sentiment_value'])
# for index,group in train_set_group:
#     #print(index)
#     print(group.count())

# 6.统计将句子划分为字符长度后，句子的长度和平均长度
def statistic_char_long(trainSet):
    print(trainSet.char.values.tolist())

    all_char = []
    for eachline in trainSet.char.values.tolist():
        all_char.append(eval(eachline))
    print(all_char)

    char_df = pd.DataFrame(all_char)
    print(char_df.describe())

    max = 0
    min = 10000
    mean = []
    for each in all_char:
        if len(each) < min:
            min = len(each)
        if len(each) > max:
            max = len(each)
        mean.append(len(each))
    import numpy as np
    print(max)
    print(min)
    print(np.array(mean).mean())

                # 最长句子为181
                # 最短句子为7
                # 平均句子长为40

# 7.




