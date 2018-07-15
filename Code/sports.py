import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation
import csv
import string
from collections import Counter
from tqdm import tqdm
import collections, re
import random
from random import randint
from sklearn.metrics import average_precision_score
import pandas as pd
from scipy import misc as cv2
import glob
import tensorflow as tf
from PIL import Image
from skimage import transform
import copy
from random import shuffle
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import os
import time
import imageio


def generate_training_data(folder):
    r=0
    "Gets images for training, adds labels and returns training data"
    print("Getting images for training..")
    training_data = []
    bag=[]
    label=[]
    with tqdm(total=len(glob.glob('/my_data/'+folder+"/*.png"))) as pbar:
        for img in glob.glob('/my_data/'+folder+"/*.png"):
            temp=[]
            #if r>=25:
            #    break
            if "football" in img:
                tr=[1,0,0,0]
                n= cv2.imread(img)
            elif "cricket" in img:
                tr=[0,1,0,0]
                n= cv2.imread(img)
            elif "basketball" in img:
                tr=[0,0,1,0]
                n= cv2.imread(img)
            elif "wwe" in img:
                tr=[0,0,0,1]
                n= cv2.imread(img)
            else:
                n= cv2.imread(img)
                tr=[0]
            temp.append(n)
            temp.append(tr)
            bag.append(temp)
            pbar.update(1)
            r+=1
    return bag

trainingdata=[]
trainingdata+=generate_training_data("Football")
trainingdata+=generate_training_data("Cricket")
trainingdata+=generate_training_data("Basketball")
trainingdata+=generate_training_data("WWE")
#print(len(trainingdata))

shuffle(trainingdata)

#print(trainingdata[0][1])
X=[]
y=[]
print("Resizing images")
with tqdm(total=len(trainingdata)) as p1bar:
    for i in range(len(trainingdata)):
        x=np.array(transform.resize(trainingdata[i][0],[50,50,1]),dtype='float32')
        X.append(x)
        y.append(trainingdata[i][1])
        p1bar.update(1)
del trainingdata

X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2)

tf.reset_default_graph()
convnet=input_data(shape=[None,50,50,1],name='input')
convnet=conv_2d(convnet,32,5,activation='relu')
convnet=max_pool_2d(convnet,5)
convnet=conv_2d(convnet,64,5,activation='relu')
convnet=max_pool_2d(convnet,5)

convnet=conv_2d(convnet,32,5,activation='relu')
convnet=max_pool_2d(convnet,5)

convnet=fully_connected(convnet,128,activation='relu')
convnet=dropout(convnet,0.4)
convnet=fully_connected(convnet,4,activation='softmax')
convnet=regression(convnet,optimizer='adam',learning_rate=0.005,loss='categorical_crossentropy',name='SportsClassifier')
model=tflearn.DNN(convnet,tensorboard_dir='log',tensorboard_verbose=0)
model.fit(X_train,y_train, n_epoch=20,validation_set=(X_test,y_test), snapshot_step=20,show_metric=True,run_id='SportsClassifier')

print("Saving model..")
model.save('/output/model.tflearn')

#print(len(X))
#print(len(y))

