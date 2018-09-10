# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/7

# tok_raw = Tokenizer(num_words=max_features)     # 标点符号自动过滤
#     tok_raw.fit_on_texts(all_word)                  # 必须是一维数组
#     return tok_raw
#
# def updata_tokenizer(text_list,tok_raw):
#     tok_raw.fit_on_texts(text_list)                  # 必须是一维数组
#     return tok_raw



# from  keras.preprocessing.text import Tokenizer
# tok_raw = Tokenizer(num_words=100)
# #print(tok_raw.word_index)
# tok_raw.fit_on_texts(['hahah','cao'])
# print(tok_raw.word_index)
# tok_raw.fit_on_texts(['ns'])
# print(tok_raw.word_index)



import pandas as pd

data = pd.DataFrame({'sheng':[1,2,3],'wei':[1,2,3]})

print(data)


data = data.iloc[0:1]
print(data)

