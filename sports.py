import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing, cross_validation
import csv
import string
from tqdm import tqdm
import random
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


path="./model.tflearn.meta"        #base path of all classes' folders


def generate_training_data(folder):
    r=0
    "Gets images for training, adds labels and returns training data"
    print("Getting images for training..")
    training_data = []
    bag=[]
    label=[]
    with tqdm(total=len(glob.glob(folder+"/*.png"))) as pbar:
        for img in glob.glob(folder+"/*.png"):
            temp=[]
            #if r>=30:
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

if os.path.exists(path):
    print("Model found, loading it..")
    tf.reset_default_graph()
    convnet=input_data(shape=[None,32,32,3],name='input')
    convnet=conv_2d(convnet,32,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,64,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,128,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,256,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,512,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,256,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,128,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,64,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    
    
    #convnet=dropout(max_2,0.8)
    convnet=fully_connected(convnet,4,activation='softmax')
    convnet=regression(convnet,optimizer='adam',learning_rate=0.005,loss='categorical_crossentropy',name='ScreenshotClassifier')
    model=tflearn.DNN(convnet,tensorboard_dir='log',tensorboard_verbose=3)
    model.load('./model.tflearn')
    bag=generate_training_data("test")
    random.shuffle(bag)

    i=0
    data=[]
    labels=[]
    print("Getting test data..")
    for i in range(len(bag)):       #sepearting features and labels
        data.append(bag[i][0])      #just images for test data, no labels
    del bag

    i=0
    X=[]
    print("Resizing images")
    with tqdm(total=len(data)) as p1bar:
        for i in range(len(data)):
            #if i>=90:
            #    break
            x=np.array(transform.resize(data[i],[32,32,3]),dtype='float32')
            X.append(x)
            p1bar.update(1)
    X_test=X                     #for feeding to NN for predicting label
    real_data=copy.deepcopy(X_test)      #for displaying images in testing
    j=len(X_test)
    m=j
    s=0


    ot=0
    fb=0
    st=0
    cod=0
    gm=0
    yt=0
    imgpath="/home/asim/Desktop/Python/MiniProject/SportsDetection"        #base path of all classes' folders
    print("Predicting on test set..")

    with tqdm(total=m) as p1bar:
        while j>=0:
            i=0
            for r in (X_test):
                ts = time.time()
                img_data=r
                orig=img_data
                model_out=model.predict([img_data])[0]
                if np.argmax(model_out) ==0:
                    if os.path.exists(imgpath+"/fotball"):
                        str_label='Football'
                        ts=str(ts)
                        ts=ts.replace('.','_')
                        p="/football/"+ts+".png"
                        imageio.imwrite(imgpath+p, data[s])
                        """ else:
                        os.makedirs(imgpath+"/football")
                        str_label='Football'
                        ts=str(ts)
                        ts=ts.replace('.','_')
                        p="/football/"+ts+".png"
                        imageio.imwrite(imgpath+p, data[s]) """
                        fb+=1
                elif np.argmax(model_out) ==1:
                    if os.path.exists(imgpath+"/cricket"):
                        str_label='Cricket'
                        ts=str(ts)
                        ts=ts.replace('.','_')
                        p="/cricket/"+ts+".png"
                        imageio.imwrite(imgpath+p, data[s])
                        """ else:
                        os.makedirs(imgpath+"/cricket")
                        str_label='Cricket'
                        ts=str(ts)
                        ts=ts.replace('.','_')
                        p="/cricket/".format(ts)+".png"
                        imageio.imwrite(imgpath+p, data[s]) """
                        yt+=1
                elif np.argmax(model_out) ==2:
                    if os.path.exists(imgpath+"/basketball"):
                        str_label='Basketball'
                        ts=str(ts)
                        ts=ts.replace('.','_')
                        p="/basketball/"+ts+".png"
                        imageio.imwrite(imgpath+p, data[s])
                        """ else:
                        os.makedirs(imgpath+"/basketball")
                        str_label='Basketball'
                        ts=str(ts)
                        ts=ts.replace('.','_')
                        p="/basketball/"+ts+".png"
                        imageio.imwrite(imgpath+p, data[s]) """
                        st+=1
                elif np.argmax(model_out) ==3:
                    if os.path.exists(imgpath+"/wwe"):
                        str_label='WWE'
                        ts=str(ts)
                        ts=ts.replace('.','_')
                        p="/wwe/"+ts+".png"
                        imageio.imwrite(imgpath+p, data[s])
                        """ else:
                        os.makedirs(imgpath+"/wwe")
                        str_label='WWE'
                        ts=str(ts)
                        ts=ts.replace('.','_')
                        p="/wwe/"+ts+".png"
                        imageio.imwrite(imgpath+p, data[s]) """
                        gm+=1
                i+=1
                j-=1
                s+=1
                p1bar.update(1)
                if(j<=0):
                    j=-2
                    break
                if(i>=15):        #cause plt figure size
                    break
            if(j>=0):
                X_test=copy.deepcopy(X_test[15:])
        #plt.show()
    
else:
    print("No model found")
    trainingdata=[]
    trainingdata+=generate_training_data("TrainingData/Football")
    trainingdata+=generate_training_data("TrainingData/Cricket")
    trainingdata+=generate_training_data("TrainingData/Basketball")
    trainingdata+=generate_training_data("TrainingData/WWE")
    print(len(trainingdata))

    random.shuffle(trainingdata)

    #print(trainingdata[0][1])
    X=[]
    y=[]
    print("Resizing images")
    with tqdm(total=len(trainingdata)) as p1bar:
        for i in range(len(trainingdata)):
            x=np.array(transform.resize(trainingdata[i][0],[32,32,3]),dtype='float32')
            X.append(x)
            y.append(trainingdata[i][1])
            p1bar.update(1)
    del trainingdata

    X_train, X_test, y_train, y_test=cross_validation.train_test_split(X,y,test_size=0.2)

    tf.reset_default_graph()
    
    convnet=input_data(shape=[None,32,32,3],name='input')
    convnet=conv_2d(convnet,32,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,64,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,128,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,256,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,512,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,256,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,128,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    convnet=conv_2d(convnet,64,5,activation='relu')
    convnet=max_pool_2d(convnet,5)
    
    
    #convnet=dropout(convnet,0.2)
    convnet=fully_connected(convnet,4,activation='softmax')
    convnet=regression(convnet,optimizer='adam',learning_rate=0.005,loss='categorical_crossentropy',name='SportsClassifier')
    model=tflearn.DNN(convnet,tensorboard_dir='log',tensorboard_verbose=0)

    model.fit(X_train,y_train, n_epoch=40,validation_set=(X_test,y_test), snapshot_step=20,show_metric=True,run_id='SportsClassifier')

    print("Saving model..")
    model.save('model.tflearn')



    
