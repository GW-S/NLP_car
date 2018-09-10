# author:sheng.Gw
# -*- coding: utf-8 -*-
# @Date :  2018/9/9
import pandas as pd
result = pd.read_csv('result1.csv')

result['sentiment_value'] = [int(each) for each in result['sentiment_value'].tolist()]

result.to_csv('result2.csv',index=False)