with open('scentence.txt','r',encoding='utf-8') as f:
    data = [line.split('\t') for line in f.read().splitlines()]
    data=data[1:]

a=0
for i in data:
    i.append(a)
    if i[0]=='':
        a+=1
for i in range(len(data)-1,0,-1):
    if data[i][0]=='':
        del data[i]

words=data

from konlpy.tag import Okt
okt=Okt()
def term(doc):
    return [doc.count(word) for word in they]
def tok(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return [''.join(t[0]) for t in okt.pos(doc, norm=True, stem=True)]

they=[]
for i in words:
    for j in okt.pos(i[0],norm=True,stem=True):
        print(j)
        they.append(j[0])
    
    

asd=[[tok(d[0]),d[1]] for d in data]

they=list(set(they))

datas=[]
for d in asd:
    datas.append(term(d[0]))
y_data=[]
for d in asd:
    if d[1]==0:
        y_data.append([1,0,0])
    elif d[1]==1:
        y_data.append([0,1,0])
    else:
        y_data.append([0,0,1])

import numpy as np
datas=np.asarray(datas)



datas=datas.astype('float32')
inputs=datas.shape[1]
y_data=np.array(y_data)

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import tensorflow as tf

model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(inputs,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation=tf.nn.softmax))

model.compile(optimizer='Adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
model.fit(datas, y_data, epochs=5)
a=input('기분을 나타내는 문장을 입력해 주세요')
asdf=np.asarray([term(tok(a))]).astype('float32')
prediction=model.predict(asdf)
prediction=np.argmax(prediction)
if prediction==0:
    print('화나')
if prediction==1:
    print('슬퍼')
if prediction==2:
    print('기뻐')

qwer=np.array([datas[0]])






