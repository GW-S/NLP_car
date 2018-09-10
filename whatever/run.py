# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/7
"""
该文件只进行情感分析
"""
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

from keras.layers import Input, Concatenate
from whatever.utils_really import read_train_set
from whatever.utils_really import read_test_set



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
'''模型训练'''
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM


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



def cnn(config):

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


def cnn_topic():
    pass

    # Inputs:定义输入的形状
    comment_seq = Input(shape =(maxlen,) ,name='Discuss_seq')
    # Embeddings layers 进行简单的词嵌入
    emb_comment = Embedding(max_features, embed_size)(comment_seq)
    # 卷积层
    convs = []
    filter_sizes = [2, 3, 4, 5, 6]

    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)

    merge = concatenate(convs, axis=1)  # concatenate的作用是将sequential压缩成一个输出
    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)
    output = Dense(units=1, activation='linear')(output)
    model = Model([comment_seq], output)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", score]) # mae的意思是：mean absolute error

    return model


# 转换格式，必须换成整数，不然不接受,这是sklearn的限制


def lstm(train_x, train_y,tokenizer):
    # 建立序列结构
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index) + 1, 200))
    model.add(LSTM(128, dropout=0.2, input_dim=1, input_length=200))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, batch_size=32, epochs=3)  # epochs 训练次数
    # 训练之后保存模型
    model.save('./files/model_file/model2.h5')
    return model






    #
    #
    #
    # '''用模型进行预测'''
    #
    #
    #
    # # 加载使用的模型
    # from keras.models import load_model
    # model = load_model('./files/model_file/model2.h5')
    #
    #
    #
    #
    #
    #
    # # 进行预测
    # predict_label = model.predict_classes(test_x)
    #
    # # 加到列表后
    # predictSet_jieba['predict_result'] = list(predict_label)
    #
    # # 转化为提交的文件格式
    # result = predictSet_jieba[['Id', 'predict_result']]
    #
    # savepath = './files/result_file/model2'
    # result.to_csv(savepath)
    #





# todo: 能不能把值与值之间的间距尽量拉开


def getTheMetricAttribute(predict,really):
    print(type(predict[0]))
    print(type(really[0]))
    #accuracy_Score = accuracy_score(really, predict)





    #print("这是准确率",accuracy_Score)


config ={
    'try_time':1,  ## 是否是在进行测试，1代表在用少量数据进行测试，0代表用全部数据进行训练
    'max_features':80000,  ## 词汇量
    'maxlen':200,  ## 最大长度
    'embed_size':200,  ## emb 长度
    'stopword_path':'/Users/sheng/PycharmProjects/NLP_car/dependence_file/new_stopword.txt',
    'batch_size':128,
    'epochs':1,
}



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


def sentiment_analystic(config):
    # 读取训练集
    train_set = read_train_set()
    # 读取测试集
    test_set = read_test_set()

    trainSet_jieba_noStopword = preprocessor_train_and_test(config, train_set)
    testSet_jieba_noStopword = remove_stopword(test_set,config['stopword_path'])

    # 构建词袋模型
    train_tokenizer = construct_tokenizer(trainSet_jieba_noStopword,config)
    train_tokenizer = updata_sheng_tokenizer(test_set['content'].tolist(), train_tokenizer)
    print(len(train_tokenizer.word_index))

    # 进行padding
    trainSet_jieba_noStopword = use_padding(trainSet_jieba_noStopword, train_tokenizer,config)
    testSet_jieba_noStopword = use_padding(testSet_jieba_noStopword, train_tokenizer,config)

    # 构建情感分析模型   # todo:一个机器重要的技巧，dataframd出来的一般是series,很多时候其他系统是不认可这个的，需要to_list
    train_x, test_x, train_y, test_y = train_test_split(trainSet_jieba_noStopword['pad_sequence'].tolist(),
                                                        [float(each) for each in
                                                         trainSet_jieba_noStopword['sentiment_value'].tolist()])
    # 提前结束训练
    early_stopping = EarlyStopping(monitor="val_score", mode="min", patience=2)
    callbacks_list = [early_stopping]

    # run_model
    model = cnn(config)

    model.fit(train_x, train_y,
              validation_split=0.1,
              batch_size=config['batch_size'],
              epochs=config['epochs'],
              shuffle=True,
              callbacks=callbacks_list)
    preds = model.predict(test_x)

    preds = model.predict(testSet_jieba_noStopword['pad_sequence'].tolist())

    # 写文件
    testSet_jieba_noStopword['sentiment_value'] = preds

    return testSet_jieba_noStopword


if __name__ == '__main__':



    sentiment_result:pd.DataFrame = sentiment_analystic(config)









    # 进行主题提取
    def topic_extract(trainSet_jieba_noStopword):

        # 读取训练集
        train_set = read_train_set()
        # 读取测试集
        test_set = read_test_set()

        trainSet_jieba_noStopword = preprocessor_train_and_test(config, train_set)
        testSet_jieba_noStopword = remove_stopword(config, test_set)

        # 构建词袋模型
        train_tokenizer = construct_tokenizer(trainSet_jieba_noStopword,config)
        train_tokenizer = updata_sheng_tokenizer(test_set['content'].tolist(), train_tokenizer)
        print(len(train_tokenizer.word_index))

        # 进行padding
        trainSet_jieba_noStopword = use_padding(trainSet_jieba_noStopword, train_tokenizer)
        testSet_jieba_noStopword = use_padding(testSet_jieba_noStopword, train_tokenizer)

        # 得到10个主题
        topic = list(set(trainSet_jieba_noStopword.subject.values.tolist()))
        # 对这10个主题进行编号
        train_tokenizer = updata_sheng_tokenizer(topic,train_tokenizer)

        # 对十个主题进行分类
        trainSet_jieba_noStopword.apply(tokenizer_topic,axis=1,args=(topic,))





        # todo:要不要进行规约化？
        # todo:什么是embedding
        # todo:将split划分在最后一个部分

        # 然后放到模型中训练

        # 构建词袋模型
        train_tokenizer = construct_tokenizer(trainSet_jieba_noStopword)
        trainSet_jieba_noStopword = use_padding(trainSet_jieba_noStopword, train_tokenizer)

        # 进行情感分析

        # 构建情感分析模型   # todo:一个机器重要的技巧，dataframd出来的一般是series,很多时候其他系统是不认可这个的，需要to_list
        train_x, test_x, train_y, test_y = train_test_split(trainSet_jieba_noStopword['pad_sequence'].tolist(),
                                                            [float(each) for each in
                                                             trainSet_jieba['sentiment_value'].tolist()])
        # 提前结束训练
        early_stopping = EarlyStopping(monitor="val_score", mode="min", patience=2)
        callbacks_list = [early_stopping]

        # run_model
        model = cnn()

        model.fit(train_x, train_y,
                  validation_split=0.1,
                  batch_size=batch_size,
                  epochs=epochs,
                  shuffle=True,
                  callbacks=callbacks_list)

        preds = model.predict(test_x)


        # test
        # 对训练集进行分词
        test_jieba = jieba_pd_DataFrame(test_set)
        # 去停用词
        testSet_jieba_noStopword = remove_stopword(test_jieba, stopword_path)
        # 分析test文件
        testSet_jieba_noStopword = use_padding(testSet_jieba_noStopword, train_tokenizer)
        #
        preds = model.predict(testSet_jieba_noStopword['pad_sequence'].tolist())



        # 写文件
        testSet_jieba_noStopword['subject'] = giveback_topic(preds.tolist(),topic)


        return testSet_jieba_noStopword


    # topic_result = topic_extract(trainSet_jieba_noStopword)
    #
    #


















