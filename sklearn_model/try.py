# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/10



from sklearn.feature_extraction.text import CountVectorizer

cop = ['我 不 太 好','你 呢']

vectorizer = CountVectorizer(analyzer='char')
print(vectorizer.fit_transform(cop))

print(vectorizer.vocabulary_)

