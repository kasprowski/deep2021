'''
Prepares artificial data for classification

prepare_samples_areas():
- vertical dashes in top-left area for label=0
- vertical dashes in bottom-right area for label=1

prepare_samples_colors():
- white dashes for label=1
- grey dashes for label=0

prepare_samples_direction():
- horizontal dashes for label=1
- vertical dashes for label=0

@author: pawel@kasprowski.pl
'''

import numpy as np
import random
import cv2
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer


length = 50 #size of images
size = 100 #number of samples for each class


def prepare_samples_areas():
    samplesIMG = []
    labels = []
    dashes = 50 #number of dashes
    
    #images with lines NW
    for i in range(size):
        sample = np.zeros((length,length))
        for _ in range(dashes):
            x = random.randrange(0,length/2)
            y = random.randrange(0,length/2)
            sample[x:x+6,y:y+1]=255
        samplesIMG.append(sample)
        labels.append(1)
        if i==0: cv2.imwrite("v.jpg",sample)
    
    #images with lines SE
    for i in range(size):
        sample = np.zeros((length,length))
        for _ in range(dashes):
            x = random.randrange(length/2,length)
            y = random.randrange(length/2,length)
            sample[x:x+6,y:y+1]=255
        samplesIMG.append(sample)
        labels.append(0)
        if i==0: cv2.imwrite("h.jpg",sample)

    samplesIMG = np.array(samplesIMG)
    labels = np.array(labels)

    #one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels,2)
    return samplesIMG,labels

def prepare_samples_colors():
    samplesIMG = []
    labels = []
    dashes = 50 #number of dashes
    
    #images with white lines
    for i in range(size):
        sample = np.zeros((length,length))
        for _ in range(dashes):
            x = random.randrange(0,length)
            y = random.randrange(0,length)
            sample[x:x+6,y:y+1]=255
        samplesIMG.append(sample)
        labels.append(1)
        if i==0: cv2.imwrite("v.jpg",sample)
    
    #images with gray lines
    for i in range(size):
        sample = np.zeros((length,length))
        for _ in range(dashes):
            x = random.randrange(0,length)
            y = random.randrange(0,length)
            sample[x:x+6,y:y+1]=155
        samplesIMG.append(sample)
        labels.append(0)
        if i==0: cv2.imwrite("h.jpg",sample)

    samplesIMG = np.array(samplesIMG)
    labels = np.array(labels)

    #one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels,2)
    return samplesIMG,labels


def prepare_samples_direction():
    samplesIMG = []
    labels = []
    dashes = 100 #number of dashes
    
    #images with horizontal lines
    for i in range(size):
        sample = np.zeros((length,length))
        for _ in range(dashes):
            x = random.randrange(0,length)
            y = random.randrange(0,length)
            sample[x:x+6,y:y+1]=255
        samplesIMG.append(sample)
        labels.append(1)
        if i==0: cv2.imwrite("h.jpg",sample)
    
    #images with vertical lines
    for i in range(size):
        sample = np.zeros((length,length))
        for _ in range(dashes):
            x = random.randrange(0,length)
            y = random.randrange(0,length)
            sample[x:x+1,y:y+6]=255
        samplesIMG.append(sample)
        labels.append(0)
        if i==0: cv2.imwrite("v.jpg",sample)

    samplesIMG = np.array(samplesIMG)
    labels = np.array(labels)

    #one-hot encoding
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels,2)
    return samplesIMG,labels
