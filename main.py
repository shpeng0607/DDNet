import numpy as np
import math
import random
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import glob
import gc
from utils import *
from tqdm import tqdm
import pickle
import itertools

from tensorflow.keras.optimizers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import tensorflow.keras
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

random.seed(123)

class Config():
    def __init__(self):
        self.frame_l = 32 # the length of frames
        self.joint_n = 22 # the number of joints
        self.joint_d = 3 # the dimension of joints
        self.clc_coarse = 14 # the number of coarse class
        self.clc_fine = 28 # the number of fine-grained class
        self.feat_d = 231
        self.filters = 64
        self.data_dir = 'SHREC/'
C = Config()

def poses_diff(x):
    H, W = x.get_shape()[1],x.get_shape()[2]
    x = tf.subtract(x[:,1:,...],x[:,:-1,...])
    x = tf.image.resize(x,size=[H,W])
    return x

def pose_motion(P,frame_l):
    # vslow
    P_vslow = concatenate([P,P],axis=1)
    P_diff_vslow = Lambda(lambda x: poses_diff(x))(P_vslow)
    P_diff_vslow = Reshape((int(2*frame_l),-1))(P_diff_vslow)

    P_diff_slow = Lambda(lambda x: poses_diff(x))(P)
    P_diff_slow = Reshape((frame_l,-1))(P_diff_slow)
    P_fast = Lambda(lambda x: x[:,::2,...])(P)
    P_diff_fast = Lambda(lambda x: poses_diff(x))(P_fast)
    P_diff_fast = Reshape((int(frame_l/2),-1))(P_diff_fast)
    return P_diff_vslow,P_diff_slow,P_diff_fast

def c1D(x,filters,kernel):
    x = Conv1D(filters, kernel_size=kernel,padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def block(x,filters):
    x = c1D(x,filters,3)
    x = c1D(x,filters,3)
    return x

def d1D(x,filters):
    x = Dense(filters,use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def build_FM(frame_l=32,joint_n=22,joint_d=2,feat_d=231,filters=16):
    W = Input(shape=(frame_l,feat_d))
    M = Input(shape=(frame_l,feat_d))
    P = Input(shape=(frame_l,joint_n,joint_d))

    diff_vslow,diff_slow,diff_fast = pose_motion(P,frame_l) # vslow

    x = c1D(M,filters*2,1)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x,filters,3)
    x = SpatialDropout1D(0.1)(x)
    x = c1D(x,filters,1)
    x = MaxPooling1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    xw = c1D(W,filters*2,1)
    xw = SpatialDropout1D(0.1)(xw)
    xw = c1D(xw,filters,3)
    xw = SpatialDropout1D(0.1)(xw)
    xw = c1D(xw,filters,1)
    xw = MaxPooling1D(2)(xw)
    xw = SpatialDropout1D(0.1)(xw)

    # vslow
    x_d_vslow = c1D(diff_vslow,filters*2,1)
    x_d_vslow = SpatialDropout1D(0.1)(x_d_vslow)
    x_d_vslow = c1D(x_d_vslow,filters,3)
    x_d_vslow = SpatialDropout1D(0.1)(x_d_vslow)
    x_d_vslow = c1D(x_d_vslow,filters,1)
    x_d_vslow = MaxPool1D(4)(x_d_vslow)
    x_d_vslow = SpatialDropout1D(0.1)(x_d_vslow)

    x_d_slow = c1D(diff_slow,filters*2,1)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,3)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)
    x_d_slow = c1D(x_d_slow,filters,1)
    x_d_slow = MaxPool1D(2)(x_d_slow)
    x_d_slow = SpatialDropout1D(0.1)(x_d_slow)

    x_d_fast = c1D(diff_fast,filters*2,1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,3)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)
    x_d_fast = c1D(x_d_fast,filters,1)
    x_d_fast = SpatialDropout1D(0.1)(x_d_fast)

    x = concatenate([xw,x,x_d_vslow,x_d_slow,x_d_fast]) # vslow
    x = block(x,filters*2)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x,filters*4)
    x = MaxPool1D(2)(x)
    x = SpatialDropout1D(0.1)(x)

    x = block(x,filters*8)
    x = SpatialDropout1D(0.1)(x)

    return Model(inputs=[W,M,P],outputs=x)

def build_DD_Net(frame_l=32,joint_n=22,joint_d=3,feat_d=66,clc_num=14,filters=16):
    W = Input(name='W', shape=(frame_l,feat_d))
    M = Input(name='M', shape=(frame_l,feat_d))
    P = Input(name='P', shape=(frame_l,joint_n,joint_d))

    FM = build_FM(frame_l,joint_n,joint_d,feat_d,filters)

    x = FM([W,M,P])

    x = GlobalMaxPool1D()(x)

    x = d1D(x,128)
    x = Dropout(0.3)(x)
    x = d1D(x,128)
    x = Dropout(0.3)(x)
    x = Dense(clc_num, activation='softmax')(x)

    ######################Self-supervised part
    model = Model(inputs=[W,M,P],outputs=x)
    return model

DD_Net = build_DD_Net(C.frame_l,C.joint_n,C.joint_d,C.feat_d,C.clc_fine,C.filters)
DD_Net.summary()

Train = pickle.load(open(C.data_dir+"train.pkl", "rb"))
Test = pickle.load(open(C.data_dir+"test.pkl", "rb"))
X_ = []
X_0 = []
X_1 = []
Y = []
for i in tqdm(range(len(Train['pose']))):
    p = np.copy(Train['pose'][i]).reshape([-1,22,3])
    p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)
    p = normlize_range(p)

    label = np.zeros(C.clc_fine)
    label[Train['fine_label'][i]-1] = 1

    M = get_CG(p,C)
    W = get_CGW(p,C)

    X_.append(W)
    X_0.append(M)
    X_1.append(p)
    Y.append(label)

X_ = np.stack(X_)
X_0 = np.stack(X_0)
X_1 = np.stack(X_1)
Y = np.stack(Y)

X_test = []
X_test_0 = []
X_test_1 = []
Y_test = []
for i in tqdm(range(len(Test['pose']))):
    p = np.copy(Test['pose'][i]).reshape([-1,22,3])
    p = zoom(p,target_l=C.frame_l,joints_num=C.joint_n,joints_dim=C.joint_d)
    p = normlize_range(p)

    label = np.zeros(C.clc_fine)
    label[Test['fine_label'][i]-1] = 1

    M = get_CG(p,C)
    W = get_CGW(p,C)

    X_test.append(W)
    X_test_0.append(M)
    X_test_1.append(p)
    Y_test.append(label)

X_test = np.stack(X_test)
X_test_0 = np.stack(X_test_0)
X_test_1 = np.stack(X_test_1)
Y_test = np.stack(Y_test)

lr = 1e-3
DD_Net.compile(loss="categorical_crossentropy",optimizer=tensorflow.keras.optimizers.Adam(lr),metrics=['accuracy'])
lrScheduler = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=5, cooldown=5, min_lr=5e-6)
history = DD_Net.fit([X_,X_0,X_1],Y,
                    batch_size=len(Y),
                    epochs=800,
                    verbose=True,
                    shuffle=True,
                    callbacks=[lrScheduler],
                    validation_data=([X_test,X_test_0,X_test_1],Y_test)
                    )

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

DD_Net.save_weights('weights/fine_lite.h5')


lr = 1e-5
DD_Net.compile(loss="categorical_crossentropy",optimizer=tensorflow.keras.optimizers.Adam(lr),metrics=['accuracy'])
epochs = 10
for e in range(epochs):
    print('epoch{}'.format(e))
    X_ = []
    X_0 = []
    X_1 = []
    Y = []

    for i in tqdm(range(len(Train['pose']))):

        label = np.zeros(C.clc_fine)
        label[Train['fine_label'][i]-1] = 1

        p = np.copy(Train['pose'][i]).reshape([-1,22,3])
        p = sampling_frame(p,C)

        p = normlize_range(p)
        M = get_CG(p,C)
        W = get_CGW(p,C)

        X_.append(W)
        X_0.append(M)
        X_1.append(p)
        Y.append(label)

    X_ = np.stack(X_)
    X_0 = np.stack(X_0)
    X_1 = np.stack(X_1)
    Y = np.stack(Y)


    DD_Net.fit([X_,X_0,X_1],Y,
            batch_size=len(Y),
            epochs=1,
            verbose=True,
            shuffle=True,
            validation_data=([X_test,X_test_0,X_test_1],Y_test)
            )
from sklearn.metrics import plot_confusion_matrix

Y_pred = DD_Net.predict([X_test,X_test_0,X_test_1])
cnf_matrix = plot_confusion_matrix(np.argmax(Y_test,axis=1),np.argmax(Y_pred,axis=1))
plt.figure(figsize=(10,10))
plt.imshow(cnf_matrix)
plt.show()