# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/10

import pandas as pd
from whatever.utils_really import read_train_set
from whatever.utils_really import read_test_set

from whatever.model_utils import preprocessor_train_and_test
from whatever.model_utils import construct_tokenizer
from whatever.model_utils import updata_sheng_tokenizer
from whatever.model_utils import use_padding
from whatever.model_utils import cnn_sentiment

from whatever.model_utils import Lstm_sentiment
from whatever.model_utils import Lstm_topic
from whatever.model_utils import giveback_topic

from whatever.model_utils import convert_topic


from whatever.model_utils import tokenizer_topic

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from keras.models import load_model



config ={
    'try_time':0,  ## 是否是在进行测试，1代表在用少量数据进行测试，0代表用全部数据进行训练
    'max_features':80000,  ## 词汇量
    'maxlen':200,  ## 最大长度
    'embed_size':200,  ## emb 长度
    'stopword_path':'/Users/sheng/PycharmProjects/NLP_car/dependence_file/new_stopword.txt',
    'batch_size':128,
    'epochs':1,
}






def sentiment_analystic(config):
    # 读取训练集
    train_set = read_train_set()
    # 读取测试集
    test_set = read_test_set()
    # 数据预处理
    trainSet_jieba_noStopword = preprocessor_train_and_test(config, train_set)
    testSet_jieba_noStopword =  preprocessor_train_and_test(config, test_set)
    # 构建词袋模型
    train_tokenizer = construct_tokenizer(trainSet_jieba_noStopword,config)
    train_tokenizer = updata_sheng_tokenizer(testSet_jieba_noStopword, train_tokenizer)
    # 进行padding
    trainSet_jieba_noStopword = use_padding(trainSet_jieba_noStopword, train_tokenizer,config)
    testSet_jieba_noStopword  = use_padding(testSet_jieba_noStopword,  train_tokenizer,config)
    #
    print(testSet_jieba_noStopword.iloc[0])
    print(trainSet_jieba_noStopword.iloc[0])
    # 构建分类措施
    onehot_y = to_categorical(trainSet_jieba_noStopword['sentiment_value'],3)
    print(onehot_y) # 成功的分类模式
    # 构建情感分析模型   # todo:一个机器重要的技巧，dataframd出来的一般是series,很多时候其他系统是不认可这个的，需要to_list
    train_x, test_x, train_y, test_y = train_test_split(trainSet_jieba_noStopword['pad_sequence'].tolist(),onehot_y)
    # 提前结束训练
    early_stopping = EarlyStopping(monitor="val_score", mode="min", patience=2)
    callbacks_list = [early_stopping]

    Lstm_sentiment(config,train_tokenizer,train_x,train_y,test_x,test_y)
    model = load_model('lstm_sentiwords_preversion5.h5')
    print(model.predict(train_y))
    print(model.predict_classes(train_y))
    # run_model
    # model = cnn_sentiment(config)
    #
    # model.fit(train_x, train_y,
    #           validation_split=0.1,
    #           batch_size=config['batch_size'],
    #           epochs=config['epochs'],
    #           shuffle=True,
    #           callbacks=callbacks_list)
    # preds = model.predict(test_x)
    #
    print(model.predict(testSet_jieba_noStopword['pad_sequence'].tolist()))

    preds = model.predict_classes(testSet_jieba_noStopword['pad_sequence'].tolist())

    # # 写文件
    testSet_jieba_noStopword['sentiment_value'] = preds
    return testSet_jieba_noStopword

def topic_analystic(config):
    # 读取训练集
    train_set = read_train_set()
    # 读取测试集
    test_set = read_test_set()
    # 数据预处理
    trainSet_jieba_noStopword = preprocessor_train_and_test(config, train_set)
    testSet_jieba_noStopword =  preprocessor_train_and_test(config, test_set)
    # 构建词袋模型
    train_tokenizer = construct_tokenizer(trainSet_jieba_noStopword,config)
    train_tokenizer = updata_sheng_tokenizer(testSet_jieba_noStopword, train_tokenizer)
    # 进行padding
    trainSet_jieba_noStopword = use_padding(trainSet_jieba_noStopword, train_tokenizer,config)
    testSet_jieba_noStopword  = use_padding(testSet_jieba_noStopword,  train_tokenizer,config)
    #
    print(testSet_jieba_noStopword.iloc[0])
    print(trainSet_jieba_noStopword.iloc[0])


    # 构建分类措施


    # 得到10个主题
    topic = list(set(trainSet_jieba_noStopword.subject.values.tolist()))

    # 对十个主题进行分类
    trainSet_jieba_noStopword = trainSet_jieba_noStopword.apply(tokenizer_topic, axis=1, args=(topic,))

    #
    onehot_y = to_categorical(trainSet_jieba_noStopword["subject_id"],10)
    print(onehot_y) # 成功的分类模式

    # 构建主题模型   # todo:一个机器重要的技巧，dataframd出来的一般是series,很多时候其他系统是不认可这个的，需要to_list
    train_x, test_x, train_y, test_y = train_test_split(trainSet_jieba_noStopword['pad_sequence'].tolist(),onehot_y)

    # 提前结束训练
    early_stopping = EarlyStopping(monitor="val_score", mode="min", patience=2)
    callbacks_list = [early_stopping]

    Lstm_topic(train_tokenizer,train_x,train_y,test_x,test_y)
    model = load_model('lstm_topic.h5')
    print(model.predict(train_y))
    print(model.predict_classes(train_y))

    # run_model
    # model = cnn_sentiment(config)
    #
    # model.fit(train_x, train_y,
    #           validation_split=0.1,
    #           batch_size=config['batch_size'],
    #           epochs=config['epochs'],
    #           shuffle=True,
    #           callbacks=callbacks_list)
    # preds = model.predict(test_x)
    #

    print(model.predict(testSet_jieba_noStopword['pad_sequence'].tolist()))

    preds = model.predict_classes(testSet_jieba_noStopword['pad_sequence'].tolist())

    print(preds)

    preds = convert_topic(preds.tolist(),topic)

    # # 写文件
    testSet_jieba_noStopword['topic_value'] = preds

    return testSet_jieba_noStopword


if __name__ == '__main__':
    sentiment_result:pd.DataFrame = sentiment_analystic(config)
    topic_result:pd.DataFrame = topic_analystic(config)



    def merge_two_model(sentiment_result,topic_result):
        # 融合两个模型
        print(sentiment_result.iloc[0])
        print(topic_result.iloc[0])

        sentiment_result.to_csv("sentiment_result_look_up.csv")
        topic_result.to_csv("topic_result_look_up.csv")


        add_result = topic_result[['content_id','topic_value']]
        add_result['sentiment_value'] = sentiment_result['sentiment_value']

        add_result.to_csv('result',index =False)

        back_result = []
        with open('result') as result:
            all_result = result.readlines()
            for each in all_result:
                each  = each.strip() + ',' + '\n'
                back_result.append(each)

        with open('result1','w') as result1:
            result1.writelines(back_result)

    merge_two_model(sentiment_result,topic_result)
