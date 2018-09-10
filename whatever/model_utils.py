# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/10


import pandas as pd
import jieba
import numpy as np
import pandas  as pd
from  keras.preprocessing.text import Tokenizer
from  keras.preprocessing.sequence import pad_sequences
from  keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
import jieba
import jieba.posseg
import jieba.analyse
import codecs

from keras.models import Sequential,load_model



# 对分词结果进行切分，并去除停用词
def splitWord(query, stopwords):
    wordList = jieba.cut(query)
    num = 0
    result = ''
    for word in wordList:
        word = word.rstrip()
        word = word.rstrip('"')
        if word not in stopwords:
            if num == 0:
                result = word
                num = 1
            else:
                result = result + ' ' + word
    return result
# 去停用词表
def preprocess(data):
    stopwords = {}
    #for line in codecs.open('/Users/sheng/Downloads/yunyi-master/data/stop.txt','r','utf-8'):
    for line in codecs.open('/home/guowei/scenic_spot_analysistic/CNN/new_stopword.txt', 'r', 'utf-8'):
        stopwords[line.rstrip()]=1
    # 很发人省醒的写法
    data['doc'] = data['Discuss'].map(lambda x:splitWord(x,stopwords))
    return data;

def jieba_eachLine(X):
    # X.value数据类型为numpy,所以要化为list
    thisLine  =list(X.values)
    # 对当前句子进行分词
    this_Jieba = jieba.lcut(thisLine[1].strip())            # 1控制行数
    # 在当前series后追加一列
    X = X.append(pd.Series({"jieba":this_Jieba}))           # jieba控制写的列数，需要修改
    return X

def jieba_pd_DataFrame(trainSet:pd.DataFrame):
    # 保证不会对原始trainSet的存储进行修改
    trainSet_jieba = trainSet.copy()
    # 对jieba句子应用以上结巴分词函数
    trainSet_jieba = trainSet_jieba.apply(jieba_eachLine,axis=1)
    return trainSet_jieba
# 得到列表形式的停用词表
def getTxtFileToListByUseTxtFilePath(TxtfilePath):
    """
    # 从Txt文件中读取文件，返回该文件的列表
    #:param TxtfilePath: 路径名
    #:return: 单词列表
    """
    giveBackList = []
    with open(TxtfilePath) as f:
        for line in f.readlines():
            giveBackList.append(line.strip())
    f.close()
    return giveBackList
# 去除停用词
def delete_Stop_Word_DataFrame(X, stopWord):
    thisLine = list(X.values)
    # 获得当前词的结巴分词结果
    this_now = thisLine[-1]     # -1用来表示最后一列，因为训练集和测试集的列的名称和列数是不一样的
    wordTemporary = []
    for each in this_now:
        if each in stopWord:
            continue
        else:
            wordTemporary.append(each)
    # 在当前series后追加一列
    X = X.append(pd.Series({"jieba_remove_stopword": wordTemporary}))
    return X
'''代码片：去停用词'''
def remove_stopword(trainSet_jieba,stopword_path):
    """
    :param trainSet_jieba:  DataFrame
    :return:
    """
    stopWord = getTxtFileToListByUseTxtFilePath(stopword_path)
    trainSet_jieba_noStopword  = trainSet_jieba.apply(delete_Stop_Word_DataFrame,axis=1,args=(stopWord,))
    return trainSet_jieba_noStopword
'''用标记器对训练集和预测及进行标记'''
def tokenizer_list(input_list,tokenizer):
    # 对每一行样例用训练出来的标记器进行标记
    sequences = []
    for eachline in input_list:
        thisList = []
        thisList.extend(each[0] for each in tokenizer.texts_to_sequences(eachline) if len(each)!=0)
        sequences.append(thisList)
    return sequences

def construct_tokenizer(trainSet_jieba_noStopword:pd.DataFrame,config):
    # 文本评论
    comment_text = np.hstack([trainSet_jieba_noStopword.jieba_remove_stopword.values])      # 评论预报
    all_word = []
    for eachline in comment_text:
        all_word.extend(eachline)
    tok_raw = Tokenizer(num_words=config['max_features'])     # 标点符号自动过滤
    tok_raw.fit_on_texts(all_word)                  # 必须是一维数组
    return tok_raw

def updata_sheng_tokenizer(text_list,tok_raw):
    tok_raw.fit_on_texts(text_list)                  # 必须是一维数组
    return tok_raw
# 有意思的现象是，用updata_tokenizer就会报错，这说明import的东西里面有更新这个的
# 也可能是我定义了一个同名列表

def use_padding(trainSet_jieba_noStopword,tok_raw,config):
    trainSet_jieba_noStopword['token_sequence'] = tokenizer_list(trainSet_jieba_noStopword.jieba_remove_stopword.values,tok_raw)
    sequences_result = pad_sequences(trainSet_jieba_noStopword.token_sequence.values, maxlen=config['maxlen'], padding='post')  # 有填补，阶段等功能，填充序列
    trainSet_jieba_noStopword['pad_sequence'] = sequences_result.tolist()   # ndrray竟然不能直接给dataframe
    return trainSet_jieba_noStopword

def score(y_true, y_pred):
    return 1.0 / (1.0 + K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1)))

def preprocessor_train_and_test(config,train_set:pd.DataFrame):
    # 是否进行测试
    if config['try_time'] == 1:
        train_set = train_set.iloc[0:10]
    # 对训练集进行分词
    trainSet_jieba = jieba_pd_DataFrame(train_set)
    # 去停用词
    trainSet_jieba_noStopword = remove_stopword(trainSet_jieba, config['stopword_path'])  # 一部分词已经被去掉了
    return trainSet_jieba_noStopword

def tokenizer_topic(X, topic: list):
    # X.value数据类型为numpy,所以要化为list
    print(X)
    this_topic = X['subject']
    if this_topic in topic:
        X = X.append(pd.Series({"subject_id": topic.index(this_topic)}))  # jieba控制写的列数，需要修改
    else:
        print("wrong in tokenizer_topic")
    return X





def cnn_sentiment(config):

    # Inputs:定义输入的形状
    comment_seq = Input(shape =(config['maxlen'],) ,name='Discuss_seq')                 # (batch_size,sequence_len)
    # Embeddings layers，= 进行简单的词嵌入 =
    emb_comment = Embedding(config['max_features'], config['embed_size'])(comment_seq)  # (batch_size, sequence_len, dim_size)

    # 卷积层,主要目标是抽象出所有的文本信息
    convs = []
    filter_sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(config['maxlen'] - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)  # concatenate的作用是将sequential压缩成一个输出

    #  全连接层收束
    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)                      # 问题没有错
    output = Dense(units=1, activation='linear')(output)
    model = Model([comment_seq], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", score])
    return model
    # 是这个模型本身有问题？

def Lstm_sentiment(config,tokenizer,train_x,train_y,valid_x,valid_y):
    # 训练模型
    model = Sequential()
    # embedding层将不大于word_index+1的数转换成固定长度的向量，即生成word_index+1 * output的矩阵，每一行表示该index的向量表示
    model.add(Embedding(len(tokenizer.word_index) + 1, 128))
    model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    model.fit(train_x, train_y,
              batch_size=128,
              epochs=20,
              validation_data=(valid_x, valid_y))
    model.save('lstm_sentiwords_preversion5.h5')

def Lstm_topic(tokenizer,train_x,train_y,valid_x,valid_y):
    # 训练模型
    model = Sequential()
    # embedding层将不大于word_index+1的数转换成固定长度的向量，即生成word_index+1 * output的矩阵，每一行表示该index的向量表示
    model.add(Embedding(len(tokenizer.word_index) + 1, 128))
    model.add(LSTM(128,dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    model.fit(train_x, train_y,
              batch_size=128,
              epochs=20,
              validation_data=(valid_x, valid_y))
    model.save('lstm_topic.h5')

# 将文件转为topic
def giveback_topic(preds: list, topic):
    back = []
    for each in preds:
        back.append(topic[int(each[0])])  # 这里有一个问题，返回值很可能是float,这该怎么解决
    return back
def convert_topic(preds:list,topic):
    back = []
    for each in preds:
        back.append(topic[each])
    return back





