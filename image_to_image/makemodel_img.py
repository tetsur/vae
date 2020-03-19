import heapq
import collections
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from natsort import natsorted
from PIL import Image

traindata_img = os.listdir(r'001.Black_footed_Albatross_img')
traindata_img2 = natsorted(traindata_img)
print(traindata_img2)
trainlen = []
trainlen6 = []

"""
for i in traindata2:
    datafile = '0/' + i
    num_lines = sum(1 for line in open(datafile))
    if num_lines == 6:
        root, ext = os.path.splitext(i)
        trainlen6.append(root)
        print("aaaaaaaaa")

"""

train = []
count = 0
#  訓練とテストのデータセット用意
for f in traindata_img2:
    #root, ext = os.path.splitext(f)
    img = np.array(Image.open(
        '001.Black_footed_Albatross_img/' + f), dtype='float32')
    img = img / 255
    img = img.astype('float32')
    img = img.transpose(2, 0, 1)
    print(img.shape)
    for i in range(10):
        train.append(img)
  
    
    """
    if root in trainlen6:
        for i in range(6):
            train.append(img)
    else:
        for i in range(5):
            train.append(img)
    """
    print(count)
    count = count + 1




    

    
    #img = img.transpose(2,0,1)
train = np.array(train,dtype = 'float32')
print(train.shape)
print(train[0])
np.save("birds_img",train)

