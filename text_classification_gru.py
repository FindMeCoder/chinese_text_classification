# -*- coding: utf-8 -*-
"""
Created on Sun May  5 13:38:09 2019

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

stopwords=pd.read_csv("data/stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
stopwords=stopwords['stopword'].values

def preprocess_text(content_lines,sentences,category):
    for line in content_lines:
        try:
            segs=jieba.cut(line)
            segs=filter(lambda x:len(x)>1,segs)
            segs=filter(lambda x:x not in stopwords,segs)
            sentence=[]
            for seg in segs:
                sentence.append(seg)
            sentences.append((" ".join(sentence), category))
        except Exception:
            print(line)
            continue
        
#生成训练数据
sentences = []

preprocess_text(technology, sentences, 'technology')
preprocess_text(car, sentences, 'car')
preprocess_text(entertainment, sentences, 'entertainment')
preprocess_text(military, sentences, 'military')
preprocess_text(sports, sentences, 'sports')

from sklearn.model_selection import train_test_split
x, y = zip(*sentences)
train_data, test_data, train_target, test_target = train_test_split(x, y, random_state=1234)




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders

learn=tf.contrib.learn

FLAGS=None
MAX_DOCUMENT_LENGTH = 15
MIN_WORD_FREQUENCE = 1
EMBEDDING_SIZE = 50


# 处理词汇
vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCE)
x_train = np.array(list(vocab_processor.fit_transform(train_data)))
x_test = np.array(list(vocab_processor.transform(test_data)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

def bag_of_words_model(features,target):
    """先转成词袋模型"""
    target=tf.one_hot(target,15,1,0)
    features=encoders.bow_encoder(features,vocab_size=n_words,embed_dim=EMBEDDING_SIZE)
    logits=tf.contrib.layers.fully_connected(features,15,activation_fn=None)
    train_op=tf.contrib.layers.optimize_loss(loss
                                             ,tf.contrib.framework.get_global_step()
                                             ,optimizer='Adam'
                                             ,learning_rate=0.01)
    return ({
            'class':tf.arg_max(logits,1),
            'prob':tf.nn.softmax(logits)
            },loss,train_op)
    

model_fn=bag_of_words_model
classifier=learn.SKCompat(learn.Estimator(model_fn=model_fn))


classifier.fit(x_train,y_train,steps=1000)
y_predicted=classifier.predict(x_test)['class']
score=metrics.accuracy_score(y_test,y_predicted)
print('Accuracy:{0:f}'.format(score))




#GRU分类器

def rnn_model(features, target):
	"""用RNN模型(这里用的是GRU)完成文本分类"""
	
	word_vectors = tf.contrib.layers.embed_sequence(
			features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='words')

	
	word_list = tf.unstack(word_vectors, axis=1)

	
	cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)

	_, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

	target = tf.one_hot(target, 15, 1, 0)
	logits = tf.contrib.layers.fully_connected(encoding, 15, activation_fn=None)
	loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

	# Create a training op.
	train_op = tf.contrib.layers.optimize_loss(
			loss,
			tf.contrib.framework.get_global_step(),
			optimizer='Adam',
			learning_rate=0.01)

	return ({
			'class': tf.argmax(logits, 1),
			'prob': tf.nn.softmax(logits)
	}, loss, train_op)
model_fn = rnn_model
classifier = learn.SKCompat(learn.Estimator(model_fn=model_fn))

# Train and predict
classifier.fit(x_train, y_train, steps=1000)
y_predicted = classifier.predict(x_test)['class']
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: {0:f}'.format(score))