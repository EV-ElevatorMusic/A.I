#!/usr/bin/env python
# coding: utf-8

# In[218]:


with open('scentence.txt','r',encoding='utf-8') as f:
    data = [line.split('\t') for line in f.read().splitlines()]
    data=data[1:]


# In[219]:


a=0
for i in data:
    i.append(a)
    if i[0]=='':
        a+=1
for i in range(len(data)-1,0,-1):
    if data[i][0]=='':
        del data[i]


# In[220]:


words=data


# In[221]:


from konlpy.tag import Komoran
okt=Komoran()


# In[222]:


def term(doc):
    return [doc.count(word) for word in they]
def tok(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
        return [''.join(t[0]) for t in okt.pos(doc) if t[1]!='EC']


# In[223]:


they=[]
for i in words:
    for j in okt.pos(i[0]):
        if j[1]=='EC' or j[1]=='ETM':
            pass
        else:
            they.append(j[0])
    
    


# In[224]:


asd=[[tok(d[0]),d[1]] for d in data]


# In[225]:


they=list(set(they))


# In[226]:


len(data)


# In[227]:


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


# In[228]:


import numpy as np
datas=np.asarray(datas)


# In[229]:


datas=datas.astype('float32')


# In[230]:


inputs=datas.shape[1]


# In[231]:


y_data=np.array(y_data)


# In[232]:


from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import tensorflow as tf



# In[233]:


model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(inputs,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation=tf.nn.softmax))

model.compile(optimizer='Adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# In[234]:


model.fit(datas, y_data, epochs=15
)


# In[253]:


a=input()
asdf=np.asarray([term(tok(a))]).astype('float32')
prediction=model.predict(asdf)
prediction=np.argmax(prediction)
if np.argmax(asdf[0])==0:
    print('다시입력하세요')
else:
    if prediction==0:
        print('화나')
    elif prediction==1:
        print('슬퍼')
    elif prediction==2:
        print('기뻐')


# In[147]:


tok(a)


# In[120]:


asdf=np.asarray([term(tok(a))]).astype('float32')


# In[121]:


asdf[0]


# In[70]:


for i in range(len(asdf[0])):
    if int(asdf[0][i]):
        print(they[i])


# In[71]:


prediction=model.predict(asdf)


# In[75]:


prediction


# In[ ]:





# In[76]:


qwer=np.array([datas[0]])


# In[77]:


qwer


# In[78]:


qwer.shape


# In[79]:


model.summary()


# In[254]:


model.save('model.h5')


# In[255]:


converter=tf.lite.TFLiteConverter.from_keras_model_file('model.h5')


# In[256]:


tflite_model = converter.convert()


# In[257]:


open("model.tflite", "wb").write(tflite_model)


# In[ ]:




