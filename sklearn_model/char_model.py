# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/10

# 1.用GBDT进行主题提取

from sklearn_model.util_model_split import read_train_set_char
from sklearn.model_selection import train_test_split
import numpy as np


#from sklearn_model.util_model_split import


config = {'try_time':0}


import pandas as pd
from gensim.corpora import Dictionary



if config['try_time'] == 1:
    trainSet = read_train_set_char().iloc[0:10]
else:
    trainSet = read_train_set_char()

print(trainSet.iloc[0])


# 1.特征为一个词袋，大概就：tokenizer,存在未存在，数量
# 2.特征为一个词袋，大概就:tf-idf

# 特征工程


from sklearn.feature_extraction.text import CountVectorizer

all_char = []
for eachline in trainSet.char.values.tolist():
    all_char.extend(eval(eachline))
print(all_char)
print(trainSet.char.tolist())

vectorizer = CountVectorizer(max_df= 10000,min_df=1,analyzer='char') # 必须要用char才能
vectorizer.fit(trainSet.char.values.tolist())
vectorizer.transform(trainSet.char.values.tolist())



print(vectorizer.vocabulary_)


from sklearn.feature_extraction.text import TfidfVectorizer

tf_vector = TfidfVectorizer(max_df= 10000, min_df= 1,analyzer='char',stop_words='english')



print(trainSet.char.values.tolist()[0])
tf_vector.fit(trainSet.char.values.tolist())
print(tf_vector.vocabulary_)
print(tf_vector.transform(trainSet.char.values.tolist())[1])


# 用其来提取主题
# 构建主题模型   # todo:一个机器重要的技巧，dataframd出来的一般是series,很多时候其他系统是不认可这个的，需要to_list
train_x, test_x, train_y, test_y = train_test_split(tf_vector.transform(trainSet.char.values.tolist()),trainSet['subject_id'].tolist())












# TPOT

# 针对该模型提取特征

from tpot import TPOTClassifier

tpot_config = \
    {
    'sklearn.ensemble.GradientBoostingClassifier': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05)
    },

    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
    },

    'sklearn.preprocessing.MaxAbsScaler': {
    },

    'sklearn.preprocessing.MinMaxScaler': {
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2',
                   'sigmoid'],
        'gamma': np.arange(0.0, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },

    'tpot.builtins.ZeroCount': {
    },

    # Selectors
    'sklearn.feature_selection.SelectFwe': {
        'alpha': np.arange(0, 0.05, 0.001),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 100),
        'score_func': {
            'sklearn.feature_selection.f_classif': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': np.arange(0.05, 1.01, 0.05)
    },

    'sklearn.feature_selection.RFE': {
        'step': np.arange(0.05, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    },

    'sklearn.feature_selection.SelectFromModel': {
        'threshold': np.arange(0, 1.01, 0.05),
        'estimator': {
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100],
                'criterion': ['gini', 'entropy'],
                'max_features': np.arange(0.05, 1.01, 0.05)
            }
        }
    }

}
# generations 确定子代的迭代次数
# population_size=10 是创建个体的初始数量
# offspring_size 每一代所需创造个体数
# crossover_rate 用于创造后代的个体所占的百分比
# mutation_rate 属性值随机更改的概率

# 基于遗传算法的一个东西


tpot = TPOTClassifier(generations=1, population_size=10, verbosity=2,
                       config_dict=tpot_config)
tpot.fit(train_x,train_y)
tpot.score(test_x,test_y)

tpot.export('result.py')




