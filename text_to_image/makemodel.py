from gensim.models import word2vec
import os
import numpy as np

import chainer
import chainer.distributions as D
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer.datasets import tuple_dataset
from sklearn.model_selection import train_test_split

model = word2vec.Word2Vec.load("birds_txt.model")
train = []

list0 = np.zeros(30, dtype='float32')



with open('birds.txt','r') as f:
    for line in f:
        sentenses = []
        
        line = line.rstrip()
        #line = line.rstrip('.')
        line = line.strip()
        
        words = line.split (' ')
        print(words)

        if len(words) >= 20:
            words = words[0:20]

        
        for i in words:
            v = model.wv[i]
            sentenses.extend(v)
        if len(words) < 20:
            l = 20 - len(words)
            for i in range(l):
                sentenses.extend(list0)

        sentenses = np.array(sentenses, dtype = 'float32')
        print(sentenses.shape)
        train.append(sentenses)
        
        
       
        

    


        #print(sentenses)
        
    

print('')
train = np.array(train, dtype = 'float32')
print(train.shape)
print(train[0],train[1])

np.save("birds_txt",train)

