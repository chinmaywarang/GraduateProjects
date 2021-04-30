#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import librosa
from librosa import feature
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report, confusion_matrix
import madmom
import matplotlib.pyplot as plt
import IPython
import os
import pandas as pd
import scipy.io.wavfile


# In[5]:


#importing data and seperating 
Dataset = pd.read_csv('TheWallOfSoundDataset.csv')
print(Dataset)
X = Dataset.iloc[:, 2:4].values
y = Dataset.iloc[:, 4].values
filename = Dataset.iloc[:, 0].values


# In[6]:


def mfccs(name):
    features = np.empty([1,12])
    for i in range (0,len(name)):
        path = 'Complete Dataset/'+name[i]
        x, fs = librosa.load(path,sr=44100)
        mfccs = librosa.feature.mfcc(y=x, sr=44100, n_mfcc=13) #finding mfccs
        mfccs = mfccs[1:13]  #removing first mfcc
        m = np.mean(mfccs,axis = 1)
        features = np.vstack((features,m))
    features = features[1:]
    return features


# In[7]:


def mfccs_test(name):
    features = np.empty([1,12])
    for i in range (0,len(name)):
        path = 'Test Data/'+name[i]
        x, fs = librosa.load(path,sr=44100)
        mfccs = librosa.feature.mfcc(y=x, sr=44100, n_mfcc=13) #finding mfccs
        mfccs = mfccs[1:13]  #removing first mfcc
        m = np.mean(mfccs,axis = 1)
        features = np.vstack((features,m))
    features = features[1:]
    return features


# In[8]:


#calculating the mfccs for all audio data
X_mfccs = mfccs(filename)


# In[12]:


#mapping mfccs to the dataset values
X_complete = np.concatenate((X,X_mfccs),axis=1)


# In[59]:


#creating test data 
# 1.Mother System Mix
test = Dataset.iloc[81:85,:].values
x_test = test[:,2:4]
y_test = test[:,4]
test_filename = test[:,0]
test_mfccs = mfccs(test_filename)
print(y_test)
print(test_filename)


# In[60]:


test_complete = np.concatenate((x_test,test_mfccs),axis=1)


# In[61]:


# X_train, X_test, y_train, y_test = train_test_split(X_complete, y, test_size=0.05)
Regressor = KNeighborsRegressor(n_neighbors=1)
Regressor.fit(X_complete, y)
predictions = Regressor.predict(test_complete)
print(predictions)


# In[72]:



x_1,fs  = librosa.load('Complete Dataset/'+test_filename[0],sr=44100)
h_1 = librosa.load('Impulse Responses/Impulse_1.wav',sr = 44100)
conv_1 = np.convolve(x_1,h_1[0])
x_2,fs  = librosa.load('Complete Dataset/'+test_filename[1],sr=44100)
h_2 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_2 = np.convolve(x_2,h_2[0])
x_3,fs  = librosa.load('Complete Dataset/'+test_filename[2],sr=44100)
x_4,fs  = librosa.load('Complete Dataset/'+test_filename[3],sr=44100)
h_4 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_4 = np.convolve(x_4,h_4[0])


# In[73]:


a = IPython.display.Audio(conv_1,rate=44100)
IPython.display.display(a)
b = IPython.display.Audio(conv_2,rate=44100)
IPython.display.display(b)
c = IPython.display.Audio(x_3,rate=44100)
IPython.display.display(c)
d = IPython.display.Audio(conv_4,rate=44100)
IPython.display.display(d)


# In[74]:



pad_len_1 = len(conv_1)-len(conv_2) 
pad_len_2 = len(conv_1)-len(x_3) 
conv_2_pad = np.pad(conv_2, (0, pad_len_1), 'constant')
conv_3_pad = np.pad(x_3, (0, pad_len_2), 'constant')
conv_4_pad = np.pad(conv_4, (0, pad_len_1), 'constant')
print(len(conv_1))
print(len(conv_2_pad))
print(len(conv_3_pad))
print(len(conv_4_pad))


# In[89]:


complete = conv_1*0.2+conv_2_pad+conv_3_pad*0.4+conv_4_pad
e = IPython.display.Audio(complete,rate=44100)
IPython.display.display(e)


# In[90]:


scipy.io.wavfile.write("Mother System Mix.wav", 44100, complete)


# In[91]:


# 1.Be My Baby System Mix
test_1 = Dataset.iloc[70:76,:].values
x_test_1 = test_1[:,2:4]
y_test_1 = test_1[:,4]
test_filename_1 = test_1[:,0]
test_mfccs_1 = mfccs(test_filename_1)
print(y_test_1)
print(test_filename_1)


# In[92]:


test_complete_1 = np.concatenate((x_test_1,test_mfccs_1),axis=1)


# In[93]:


Regressor = KNeighborsRegressor(n_neighbors=1)
Regressor.fit(X_complete, y)
predictions_1 = Regressor.predict(test_complete_1)
print(predictions_1)


# In[118]:


x_1_1,fs  = librosa.load('Complete Dataset/'+test_filename_1[0],sr = 44100)
h_1_1 = librosa.load('Impulse Responses/Impulse_1.wav',sr = 44100)
conv_1_1 = np.convolve(x_1_1,h_1_1[0])
x_2_1,fs  = librosa.load('Complete Dataset/'+test_filename_1[1],sr = 44100)
h_2_1 = librosa.load('Impulse Responses/Impulse_1.wav',sr = 44100)
conv_2_1 = np.convolve(x_2_1,h_2_1[0])
x_3_1,fs  = librosa.load('Complete Dataset/'+test_filename_1[2],sr = 44100)
h_3_1 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_3_1 = np.convolve(x_3_1,h_3_1[0])
x_4_1,fs  = librosa.load('Complete Dataset/'+test_filename_1[3],sr = 44100)
h_4_1 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_4_1 = np.convolve(x_4_1,h_4_1[0])
x_5_1,fs  = librosa.load('Complete Dataset/'+test_filename_1[4],sr = 44100)
h_5_1 = librosa.load('Impulse Responses/Impulse_1.wav',sr = 44100)
conv_5_1 = np.convolve(x_5_1,h_5_1[0])
x_6_1,fs  = librosa.load('Complete Dataset/'+test_filename_1[5],sr = 44100)
h_6_1 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_6_1 = np.convolve(x_6_1,h_6_1[0])


# In[119]:


a_1 = IPython.display.Audio(conv_1_1,rate=fs)
IPython.display.display(a_1)
b_1 = IPython.display.Audio(conv_2_1,rate=fs)
IPython.display.display(b_1)
c_1 = IPython.display.Audio(conv_3_1,rate=fs)
IPython.display.display(c_1)
d_1 = IPython.display.Audio(conv_4_1,rate=fs)
IPython.display.display(d_1)
e_1 = IPython.display.Audio(conv_5_1,rate=fs)
IPython.display.display(e_1)
f_1 = IPython.display.Audio(conv_6_1,rate=fs)
IPython.display.display(f_1)


# In[120]:


pad_len_1 = len(conv_5_1)-len(conv_1_1) 
pad_len_2 = len(conv_5_1)-len(conv_3_1) 
conv_11_pad = np.pad(conv_1_1, (0, pad_len_1), 'constant')
conv_21_pad = np.pad(conv_2_1, (0, pad_len_1), 'constant')
conv_31_pad = np.pad(conv_3_1, (0, pad_len_2), 'constant')
conv_41_pad = np.pad(conv_4_1, (0, pad_len_2), 'constant')
conv_61_pad = np.pad(conv_6_1, (0, pad_len_2), 'constant')
print(len(conv_11_pad))
print(len(conv_5_1))
print(len(conv_21_pad))
print(len(conv_31_pad))
print(len(conv_41_pad))
print(len(conv_61_pad))
a_x = IPython.display.Audio(conv_41_pad,rate=fs)
IPython.display.display(a_x)


# In[121]:


complete_1 = conv_11_pad*0.5+conv_21_pad*0.4+conv_31_pad*0.4+conv_41_pad*3+conv_5_1*0.05+conv_61_pad
g_1 = IPython.display.Audio(complete_1,rate=44100)
IPython.display.display(g_1)


# In[117]:


scipy.io.wavfile.write("Be My Baby System Mix.wav", 44100, complete_1)


# In[122]:


# 2. StandbyMe System Mix
test_2 = Dataset.iloc[76:81,:].values
x_test_2 = test_2[:,2:4]
y_test_2 = test_2[:,4]
test_filename_2 = test_2[:,0]
test_mfccs_2 = mfccs(test_filename_2)
print(x_test_2.shape)
print(y_test_2)
print(test_mfccs_2.shape)
print(test_filename_2)


# In[123]:


test_complete_2 = np.concatenate((x_test_2,test_mfccs_2),axis=1)


# In[124]:


Regressor = KNeighborsRegressor(n_neighbors=1)
Regressor.fit(X_complete, y)
predictions_2 = Regressor.predict(test_complete_2)
print(predictions_2)


# In[126]:


x_1_2,fs  = librosa.load('Complete Dataset/'+test_filename_2[0],sr = 44100)
h_1_2 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_1_2 = np.convolve(x_1_2,h_1_2[0])
x_2_2,fs  = librosa.load('Complete Dataset/'+test_filename_2[1],sr = 44100)
x_3_2,fs  = librosa.load('Complete Dataset/'+test_filename_2[2],sr = 44100)
h_3_2 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_3_2 = np.convolve(x_3_2,h_3_2[0])
x_4_2,fs  = librosa.load('Complete Dataset/'+test_filename_2[3],sr = 44100)
h_4_2 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_4_2 = np.convolve(x_4_2,h_4_2[0])
x_5_2,fs  = librosa.load('Complete Dataset/'+test_filename_2[4],sr = 44100)
h_5_2 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_5_2 = np.convolve(x_5_2,h_5_2[0])


# In[20]:


a_2 = IPython.display.Audio(conv_1_2,rate=fs)
IPython.display.display(a_2)
b_2 = IPython.display.Audio(x_2_2,rate=fs)
IPython.display.display(b_2)
c_2 = IPython.display.Audio(conv_3_2,rate=fs)
IPython.display.display(c_2)
d_2 = IPython.display.Audio(conv_4_2,rate=fs)
IPython.display.display(d_2)
e_2 = IPython.display.Audio(conv_5_2,rate=fs)
IPython.display.display(e_2)


# In[127]:



pad_len_12 = len(conv_4_2)-len(conv_1_2) 
pad_len_22 = len(conv_4_2)-len(x_2_2) 
pad_len_32 = len(conv_4_2)-len(conv_3_2) 
conv_12_pad = np.pad(conv_1_2, (0, pad_len_12), 'constant')
conv_22_pad = np.pad(x_2_2, (0, pad_len_22), 'constant')
conv_32_pad = np.pad(conv_3_2, (0, pad_len_32), 'constant')
conv_52_pad = np.pad(conv_5_2, (0, pad_len_12), 'constant')
print(len(conv_12_pad))
print(len(conv_4_2))
print(len(conv_22_pad))
print(len(conv_32_pad))
print(len(conv_52_pad))


# In[133]:


complete_2 = conv_12_pad*0.6+conv_22_pad*0.05+conv_32_pad*0.3+conv_52_pad*0.5+conv_4_2*0.2
g_2 = IPython.display.Audio(complete_2,rate=44100)
IPython.display.display(g_2)


# In[134]:


scipy.io.wavfile.write("Then He Kissed Me System Mix.wav", 44100, complete_2)


# In[162]:


test_dataset = pd.read_csv('Test Data/test_data_wallofsound.csv')
print(test_dataset)
x_test_4 = test_dataset.iloc[5:10, 2:4].values
y_test_4= test_dataset.iloc[5:10, 4].values
test_filename_4 = test_dataset.iloc[5:10, 0].values
print(test_x_4)
print(test_y_4)
print(test_filename_4)


# In[165]:


test_mfccs_4 = mfccs_test(test_filename_4)


# In[166]:


test_complete_4 = np.concatenate((x_test_2,test_mfccs_2),axis=1)


# In[167]:


print(test_complete_4.shape)


# In[168]:


Regressor = KNeighborsRegressor(n_neighbors=1)
Regressor.fit(X_complete, y)
predictions = Regressor.predict(test_complete_4)
print(predictions)


# In[183]:


x_1_3,fs  = librosa.load('Test Data/'+test_filename_4[0],sr = 44100)
h_1_3 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_1_3 = np.convolve(x_1_3,h_1_3[0])
x_2_3,fs  = librosa.load('Test Data/'+test_filename_4[1],sr = 44100)
h_2_3 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_2_3 = np.convolve(x_2_3,h_1_3[0])
x_3_3,fs  = librosa.load('Test Data/'+test_filename_4[2],sr = 44100)
h_3_3 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_3_3 = np.convolve(x_3_3,h_3_3[0])
x_4_3,fs  = librosa.load('Test Data/'+test_filename_4[3],sr = 44100)
h_4_3 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_4_3 = np.convolve(x_4_3,h_4_3[0])
x_5_3,fs  = librosa.load('Test Data/'+test_filename_4[4],sr = 44100)
h_5_3 = librosa.load('Impulse Responses/Impulse_0.5.wav',sr = 44100)
conv_5_3 = np.convolve(x_5_3,h_5_3[0])


# In[184]:


a_3 = IPython.display.Audio(conv_1_3,rate=fs)
IPython.display.display(a_3)
b_3 = IPython.display.Audio(conv_2_3,rate=fs)
IPython.display.display(b_3)
c_3 = IPython.display.Audio(conv_3_3,rate=fs)
IPython.display.display(c_3)
d_3 = IPython.display.Audio(conv_4_3,rate=fs)
IPython.display.display(d_3)
e_3 = IPython.display.Audio(conv_5_3,rate=fs)
IPython.display.display(e_3)
print(len(conv_1_3))
print(len(conv_2_3))
print(len(conv_3_3))
print(len(conv_4_3))
print(len(conv_5_3))


# In[190]:


pad_len_44 = len(conv_3_3)-len(conv_1_3) 
pad_len_45 = len(conv_3_3)-len(x_5_3) 
conv_13_pad = np.pad(conv_1_3, (0, pad_len_44), 'constant')
conv_23_pad = np.pad(conv_2_3, (0, pad_len_44), 'constant')
conv_43_pad = np.pad(conv_4_3, (0, pad_len_44), 'constant')
conv_53_pad = np.pad(x_5_3, (0, pad_len_45), 'constant')
print(len(conv_13_pad))
print(len(conv_3_3))
print(len(conv_23_pad))
print(len(conv_43_pad))
print(len(conv_53_pad))


# In[200]:


complete_3 = conv_13_pad*0.4+conv_23_pad*0.1+conv_3_3*0.5+conv_43_pad*0.9+conv_53_pad*0.3
g_3 = IPython.display.Audio(complete_3,rate=44100)
IPython.display.display(g_3)


# In[201]:


scipy.io.wavfile.write("Instant Karma System Mix.wav", 44100, complete_3)


# In[ ]:




