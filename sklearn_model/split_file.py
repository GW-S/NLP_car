# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/10


from sklearn_model.util_model_split import read_train_set
from sklearn_model.util_model_split import split_content
from sklearn_model.util_model_split import tokenizer_topic




# 在这里进行数据预处理


trainSet = read_train_set()

# 0.去掉文本的重复列
print(trainSet.index)
trainSet = trainSet.iloc[trainSet['content_id'].drop_duplicates().index.tolist()]

# 1.将文本切分成字符的级别
trainSet['char'] = split_content(trainSet,1)



# 3.训练结果的修正




# 得到10个主题
topic = list(set(trainSet.subject.values.tolist()))
# 对十个主题进行分类
trainSet =trainSet.apply(tokenizer_topic, axis=1, args=(topic,))



# 2.保存文件
trainSet.to_csv("char_single.csv")






