# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:04:57 2019

@author: LENOVO
"""
import jieba.analyse as analyse
import pandas as pd
df = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df = df.dropna()
lines=df.content.values.tolist()
content = "".join(lines)
#使用默认的TF_IDF分析
print("  ".join(analyse.extract_tags(content, topK=30, withWeight=False, allowPOS=())))

import jieba.analyse as analyse
import pandas as pd
df = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df = df.dropna()
lines=df.content.values.tolist()
content = "".join(lines)
print ("  ".join(analyse.extract_tags(content, topK=30, withWeight=False, allowPOS=())))

import jieba.analyse as analyse
import pandas as pd
df = pd.read_csv("./data/military_news.csv", encoding='utf-8')
df = df.dropna()
lines=df.content.values.tolist()
content = "".join(lines)

print ("  ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))))
print ("---------------------我是分割线----------------")
print( "  ".join(analyse.textrank(content, topK=20, withWeight=False, allowPOS=('ns', 'n'))))

#主题模型
import jieba.analyse as analyse
import pandas as pd
from gensim import corpora, models, similarities
import gensim


stopwords=pd.read_csv("data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values

import jieba
import pandas as pd
df = pd.read_csv("./data/technology_news.csv", encoding='utf-8')
df = df.dropna()
lines=df.content.values.tolist()

sentences=[]
for line in lines:
    try:
        segs=jieba.lcut(line)
        segs = filter(lambda x:len(x)>1, segs)
        segs = filter(lambda x:x not in stopwords, segs)
        sentence=[]
        for seg in segs:
           sentence.append(seg) 
        sentences.append(sentence)
    except Exception :
        print(line)
        continue

#词袋模型
dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(sentence) for sentence in sentences]

#LDA建模
lda=gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=20)
print(lda.print_topic(3,topn=5))
#gensim.models.ldamodel.LdaModel()
#打印所有主题
for topic in lda.print_topics(num_topics=20,num_words=8):
    print(topic[1])

 

