#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os, time
import pandas as pd
import pickle as pkl

import tensorflow as tf
import keras
from keras import layers, Input, models, Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Dropout, Flatten, Activation, concatenate, GlobalAveragePooling2D
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

import matplotlib.pyplot as plt
import time

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

from scipy.spatial import distance


# In[2]:


# Data Load
print(':: load data')
with open('WM.pkl','rb') as f:
    [fea_all, fea_all_tst, X_rs, X_tst, y, Y_tst] = pkl.load(f)

    
print(fea_all.shape, fea_all_tst.shape, len(X_rs), len(X_tst), y.shape, Y_tst.shape)


# In[3]:


# Number of each class

unique, counts = np.unique(np.where(y==1)[1], return_counts=True)
num_trn= dict(zip(unique, counts))
print("Number of Train Class", num_trn)

unique, counts = np.unique(np.where(Y_tst==1)[1], return_counts=True)
num_tst= dict(zip(unique, counts))
print("Number of Test Class", num_tst)


# # Train / Validation / Test Split

# In[4]:


#n_tst = 10000
n_clusters = 3500
n_trn = 3500
n_val= 700
#n_U=len(X_rs)-n_tst
method='batch'

all_unique, all_counts = np.unique(np.where(y==1)[1], return_counts=True)
all_k=dict(zip(all_unique, all_counts))


# In[5]:


# Standardize
scaler = StandardScaler()

fea_all=scaler.fit_transform(fea_all)
fea_all_tst=scaler.fit_transform(fea_all_tst)


# In[6]:


def _permutation(set):
    permid = np.random.permutation(len(set[0]))
    for i in range(len(set)):
        set[i] = set[i][permid]

    return set


# In[ ]:


# Random suffle array
[X_rs, y] = _permutation([X_rs, y])

if method=='random':
    
    X_val=X_rs[:n_val]
    Y_val=y[:n_val]

    X_trn = X_rs[n_val:n_trn]
    Y_trn = y[n_val:n_trn]
    
    # diversity
    unique, counts = np.unique(np.where(y[:n_trn]==1)[1], return_counts=True)
    k=dict(zip(unique, counts))
    print(k)
    d=[]
    for i, i2 in enumerate(unique):
        d.append(k[i2]/all_k[i2])
    print(":: Diversity", np.average(d))
    
elif method=='batch':

    
    start= time.time()
    # Initialize KMeans model
    kmeans = KMeans(n_clusters = n_clusters)
    kmeans.fit(fea_all)
    labels=kmeans.labels_
    cent = kmeans.cluster_centers_
    
    print('::Sum of square distance :', kmeans.inertia_)
    print("::Clustering done", time.time()-start, "\n")


    # Clutster medoid
    start2= time.time()
    query_id=[]
    acc_mode=[]
    acc_rd=[]
    diff=0
    for i in range(n_clusters):
        p=[]
        c=np.where(labels==i)[0].tolist()
        for k in range(9):
            nb=0
            for i2 in c:
                if y[i2][k]==1:
                    nb=nb+1
            p.append(nb)
        acc_mode.append(round(p[np.argmax(p)]/np.sum(p), 4))  # 최빈값 purity
        dis=distance.cdist(fea_all[c], cent[i].reshape(1,fea_all.shape[1]), 'euclidean')  # Selecting representing data
        dis_m=c[np.argmin(dis)]
        acc_rd.append(round(p[np.where(y[dis_m]==1)[0][0]]/np.sum(p), 4)) # 대표값 purity
        query_id.append(dis_m)
        if round(p[np.argmax(p)]/np.sum(p), 4) != round(p[np.where(y[dis_m]==1)[0][0]]/np.sum(p), 4):
            diff=diff+1

    print(":: Selected", time.time()-start2, "\n")
    print(":: Purity of Mode", np.average(acc_mode))  
    print(":: Purity of Representing data", np.average(acc_rd), '\n')
    
    
    #assert len(query_id)==n_clusters
    
    X=X_rs[query_id]
    Y=y[query_id]
    
    X_val=X[:n_val]
    Y_val=Y[:n_val]

    X_trn = X[n_val:]
    Y_trn = Y[n_val:]
    
    print("Query Selected")
    
    # diversity
    unique, counts = np.unique(np.where(y[query_id]==1)[1], return_counts=True)
    k=dict(zip(unique, counts))
    print(k)
    d=[]
    for i, i2 in enumerate(unique):
        d.append(k[i2]/all_k[i2])
    print(":: Diversity", np.average(d))
    
print('::Dataset size','\n',
      'Train dataset size', X_trn.shape, Y_trn.shape,'\n',
      'Val dataset size', X_val.shape, Y_val.shape,'\n',
      'Test dataset size', X_tst.shape, Y_tst.shape)


# In[ ]:


#a=['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full', 'none']
plt.bar(unique, counts)
plt.xlabel('Failure pattern')
plt.ylabel('Number of queries')


# # CNN Model

# In[ ]:


def create_model():
    dim = 64
    input_wbm_tensor = Input((dim, dim, 1))
    conv_1 = Conv2D(16, (3,3), activation='relu', padding='same')(input_wbm_tensor)
    pool_1 = MaxPool2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_1)
    conv_2 = Conv2D(32, (3,3), activation='relu', padding='same')(pool_1)
    pool_2 = MaxPool2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_2)
    conv_3 = Conv2D(64, (3,3), activation='relu', padding='same')(pool_2)
    pool_3 = MaxPool2D(pool_size=(2, 2),strides=2 ,padding='same')(conv_3)
    GAP = GlobalAveragePooling2D()(pool_3)
    dense_1 = Dense(128, activation='tanh')(GAP)
    dense_2 = Dense(128, activation='tanh')(dense_1)
    prediction = Dense(9, activation='softmax')(dense_2)

    model = models.Model(input_wbm_tensor, prediction)
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return model
model = create_model()
model.summary()


# In[ ]:


epoch=100
batch_size = 20

es = EarlyStopping(monitor='val_loss', patience=20, mode='auto', restore_best_weights=True)
history = model.fit(X_trn, Y_trn,
     validation_data=[X_val, Y_val],
     epochs=epoch,
     batch_size=batch_size,callbacks=[es]
     )


# In[ ]:


y_hat=np.argmax(model.predict(X_tst), axis=1)
y_true = np.argmax(Y_tst,axis=1)


# In[ ]:


# performance metric
print('Macro average F1 score:', f1_score(y_true, y_hat, average='macro'))
print('Micro average F1 score:', f1_score(y_true, y_hat, average='micro'))


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_true, y_hat))


# In[ ]:





# In[ ]:





# In[ ]:




