# -*- coding: utf-8 -*-
"""
Created on Fri May  3 20:27:59 2019

@author: LENOVO
"""

import jieba
import pandas as pd
df_technology = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df_technology = df_technology.dropna()

df_car = pd.read_csv("./data/car_news.csv", encoding='utf-8')
df_car = df_car.dropna()

df_entertainment = pd.read_csv("./data/entertainment_news.csv", encoding='utf-8')
df_entertainment = df_entertainment.dropna()

df_military = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df_military = df_military.dropna()

df_sports = pd.read_csv("./data/sports_news.csv", encoding='utf-8')
df_sports = df_sports.dropna()

technology = df_technology.content.values.tolist()[1000:21000]
car = df_car.content.values.tolist()[1000:21000]
entertainment = df_entertainment.content.values.tolist()[:20000]
military = df_military.content.values.tolist()[:20000]
sports = df_sports.content.values.tolist()[:20000]

#去掉停用词
stopwords=pd.read_csv("data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values

def process_text(content_lines,sentences,category):
    for line in content_lines:
        try:
            segs=jieba.cut(line)
            segs=filter(lambda x:len(x)>1,segs)
            segs=filter(lambda x:x not in stopwords,segs)
            sentence=[]
            for seg in segs:
                sentence.append(seg)
            sentences.append((" ".join(sentence),category))
        except Exception:
            print(line)
            continue
        
sentences=[]

process_text(technology,sentences,'technology')
process_text(car,sentences,'car')
process_text(entertainment,sentences,'entertainment')
process_text(military,sentences,'military')
process_text(sports,sentences,'sports')

#打乱顺序 ，一边生成可靠的训练集

import random
random.shuffle(sentences)

for sentence in sentences[:10]:
    print(sentence[0],sentence[1])

#分割训练集    
from sklearn.model_selection import train_test_split

x,y=zip(*sentences)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1234) 

#对文本进行词袋模型进行处理

from sklearn.feature_extraction.text import CountVectorizer

vec=CountVectorizer(analyzer='word',ngram_range=(1,4),max_features=4000)

vec.fit(x_train)

def get_features(x):
    vec.transform(x)
 
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(vec.transform(x_train),y_train)

#测试准确率
classifier.score(vec.transform(x_test), y_test)


#交叉验证
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score,precision_score
import numpy as np

def stratifiedkfold(x,y,clf_class,shuffle=True,n_folds=5,**kwargs):
    stratifiedk_fold=StratifiedKFold(y,n_folds=n_folds,shuffle=shuffle)
    y_pred=y[:]
    for train_index,test_index in stratifiedk_fold:
        x_train,x_test=x[train_index],x[test_index]
        y_train=y[train_index]
        clf=clf_class(**kwargs)
        clf.fit(x_train,y_train)
        y_pred[test_index]=clf.predict(x_test)
    return y_pred

NB=MultinomialNB
print(precision_score(y,stratifiedkfold(vec.transform(x),np.array(y),NB),average='macro')

#print precision_score(y, stratifiedkfold_cv(vec.transform(x),np.array(y),NB), average='macro')

        
    
    










