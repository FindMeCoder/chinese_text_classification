# -*- coding: utf-8 -*-
"""
Created on Sat May  4 12:04:18 2019

@author: LENOVO
"""
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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

class TextClassfier():
    
    def __init__(self,classifier=MultinomialNB()):
        self.classifier=classifier
        self.vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,4), max_features=20000)
        
        #self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=12000)

    def features(self,X):
        return self.vectorizer.transform(X)
    
    def fit(self,X,y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X),y)
    
    def predict(self,X):
        return self.classifier.predict(self.feature([X]))

    def score(self,X,y):
        return self.classifier.score(self.feature(X),y)

from sklearn.model_selection import train_test_split

x,y=zip(*sentences)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1234) 

 
text_classifier=TextClassfier()
text_classifier.fit(x_train,y_train)
print(text_classifier.predict('这 是 有史以来 最 大 的 一 次 军舰 演习'))
print(text_classifier.score(x_test,y_test))


from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(
    analyzer='word', # tokenise by character ngrams
    max_features=4000,  # keep the most common 1000 ngrams
)
vec.fit(x_train)

    
from sklearn.svm import SVC
svm=SVC(kernel='linear')
svm.fit(vec.transform(x_train),y_train)
svm.score(vec.transform(x_test),y_test)



import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


class TextClassifier():

    def __init__(self, classifier=SVC(kernel='linear')):
        self.classifier = classifier
        self.vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), max_features=12000)

    def features(self, X):
        return self.vectorizer.transform(X)

    def fit(self, X, y):
        self.vectorizer.fit(X)
        self.classifier.fit(self.features(X), y)

    def predict(self, x):
        return self.classifier.predict(self.features([x]))

    def score(self, X, y):
        return self.classifier.score(self.features(X), y)
text_classifier = TextClassifier()
text_classifier.fit(x_train, y_train)
print(text_classifier.predict('这 是 有史以来 最 大 的 一 次 军舰 演习'))
print(text_classifier.score(x_test, y_test))















    


