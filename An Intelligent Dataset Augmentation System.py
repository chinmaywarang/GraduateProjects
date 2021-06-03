#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().run_line_magic('cd', '"/content/drive/My Drive/MIR/Final Project/Notebook"')
get_ipython().system('pip install --upgrade librosa')
get_ipython().system('ls')


# In[3]:


import numpy as np
import pandas as pd
import librosa
import IPython
from librosa import feature
from sklearn import neighbors
from sklearn import svm
from sklearn import neural_network
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import soundfile as sf


# In[4]:


# Load amd Check the csv file

Medley_Data = pd.read_csv("testing_data.csv")
Medley_Data


# ### Helper Function: `get_file_name_and_label()` and `get_ids()`
# 
# The following helper functions have been provided for you.

# In[5]:


def get_file_name_and_label(uuid, path='Medley-solos-DB/', dataset=Medley_Data):
#Returns full file name and path from a uuid

    
    rd = dataset.loc[ (dataset['uuid4'] == uuid) ]
    file = path + 'Medley-solos-DB' + '_' + str(rd.values[0,0]) + '-'  + str(rd.values[0,2]) + '_' + rd.values[0,6] + '.wav'
    label = rd.values[0,2]
    error_label = rd.values[0,5]
    return(file, label,error_label)
                       
def get_ids(subset, path = 'Medley-solos-DB/', dataset = Medley_Data):

    
    file_array = np.array([])
    rd = dataset.loc[ (dataset['subset'] == subset) ]
    if len(rd.index) < 1:
        file_array = np.array([0])
    else:
        k = 0
        for i in range(len(rd.index)):
            file_array = np.append(file_array,rd.iloc[k,6])
            k += 1
    return(file_array)


# Divides up file names into training, validation, and test sets
tracks_train =  get_ids('training')
tracks_validate = get_ids('validation')
tracks_test = get_ids('test')

print("There are {} tracks in the training set".format(len(tracks_train)))
print("There are {} tracks in the validation set".format(len(tracks_validate)))
print("There are {} tracks in the test set".format(len(tracks_test)))


# In[6]:


def compute_features(audiofile, n_fft=2048, hop_length=512, n_mels=128, n_mfcc=20):
#Compute features for an audio file
    

    x, fs = librosa.load(audiofile);
    raw_mfcc = librosa.feature.mfcc(y=np.array(x), sr=fs, n_mfcc=n_mfcc, hop_length=hop_length, n_mels=n_mels, n_fft=n_fft)
    stripped_mfccs= np.delete(raw_mfcc, 0, 0)
    features = np.mean(stripped_mfccs, axis=1);
    features = np.append(features,np.std(stripped_mfccs, axis=1), axis=0);
    return np.array(features);


# In[7]:


def create_feature_set(id_list, path):
#Create feature set from list of input ids.

    features = [];
    labels = [];
    error_labels = [];
    for uuid in range(len(id_list)):
        name, label, error_label = get_file_name_and_label(id_list[uuid], path=path);
        feature = compute_features(name);
        features.append(feature);
        labels.append(label);
        error_labels.append(error_label);
    return(np.array(features), np.array(labels), np.array(error_labels));


# ## Part 2a: Get Mean and Standard Deviation
# 
# 
# 

# In[8]:


def get_stats(features):
#Get mean and standard deviation of each feature in a set

    
    mean_list = np.mean(features, axis=0);
    std_dev_list = np.std(features, axis=0);
    return(np.array(mean_list),np.array(std_dev_list));


# In[9]:


# Create Normal Feature Data
path='Medley-solos-DB/'

# Change this to True if you want to load prevously-computed features
load_saved_tests = True

if not load_saved_tests:
    test_set, test_labels, test_errors = create_feature_set(tracks_test,path)
    print("Test Set: " + str(test_set.shape))
    train_set, train_labels, train_errors = create_feature_set(tracks_train,path)
    print("Training Set: " + str(train_set.shape))
    validate_set, validate_labels, validate_errors = create_feature_set(tracks_validate,path)
    print("Validation Set: " + str(validate_set.shape))
    np.savetxt('test_set.csv', test_set, delimiter=',')
    np.savetxt('test_labels.csv', test_labels, delimiter=',')
    np.savetxt('test_errors.csv', test_errors, delimiter=',')
    np.savetxt('train_set.csv', train_set, delimiter=',')
    np.savetxt('train_labels.csv', train_labels, delimiter=',')
    np.savetxt('train_errors.csv', train_errors, delimiter=',')
    np.savetxt('validate_set.csv', validate_set, delimiter=',')
    np.savetxt('validate_labels.csv', validate_labels, delimiter=',')
    np.savetxt('validate_errors.csv', validate_errors, delimiter=',')
else:
    test_set = np.loadtxt('test_set.csv',delimiter=',')
    test_labels = np.loadtxt('test_labels.csv',delimiter=',')
    test_errors = np.loadtxt('test_errors.csv',delimiter=',')
    train_set = np.loadtxt('train_set.csv',delimiter=',')
    train_labels = np.loadtxt('train_labels.csv',delimiter=',')
    train_errors = np.loadtxt('train_errors.csv',delimiter=',')
    validate_set = np.loadtxt('validate_set.csv',delimiter=',')
    validate_labels = np.loadtxt('validate_labels.csv',delimiter=',')
    validate_errors = np.loadtxt('validate_errors.csv',delimiter=',')
    print("Test Set: " + str(test_set.shape))
    print("Training Set: " + str(train_set.shape))
    print("Validation Set: " + str(validate_set.shape))


# In[10]:


def create_error(samples,error_type):
#Get mean and standard deviation of each feature in a set

    samples=samples/max(abs(samples))
    if error_type=='comb':
        offset=np.random.randint(1,30)
        a1=np.pad(samples, (offset,0), 'constant', constant_values=(0,0))
        a2=np.pad(samples, (0,offset), 'constant', constant_values=(0,0))
        output=np.add(a1,a2);
    if error_type=='clip':
        output = np.where(samples>0.3, 0.0, samples)
    if error_type=='noise':
        noise = np.random.random_sample((len(samples)))
        noise = noise*(np.random.random_sample()*0.05)
        output=np.add(noise,samples);
    output=output/max(abs(output))
    #scale output to match level of original file, not sure if this is necessary.
    # Essentially "un-normalizes" the output of the function
    output=output*max(abs(samples))
    return output


# In[11]:


def audioplayer(audio_data, sr, title):
#Plots and sonifies the chord progression of an audio file

    print(title)
    a1 = IPython.display.Audio(audio_data,rate=sr)
    IPython.display.display(a1)


# ## Part 2b: Normalize Feature Sets
# 
# Using `get_stats()` find the mean and standard deviations for the training set. Then use those statistics to make all 3 data sets have a mean of 0 and standard deviation of 1.

# In[12]:


train_mean, train_std_dev = get_stats(train_set);
train_set = (train_set - train_mean.T)/train_std_dev.T
test_set = (test_set - train_mean.T)/train_std_dev.T
validate_set = (validate_set - train_mean.T)/train_std_dev.T


# Seems like the best settings are 20, 3, and 7 hidden layers, with a max imum number of iterations of 100, so I will proceed with that.

# In[13]:


def process(errors,labels,id_list,sonify=True,output=False):
#Create feature set from list of input ids.

    error_names=["none","comb","clip","noise"]
    for index in range(len(id_list)):
            name, label, error_label = get_file_name_and_label(id_list[index],path='Medley-solos-DB/');
            samples, fs = librosa.load(name);
            error_type=error_names[int(errors[index])]
            errored_samples = create_error(samples,error_type)
        if sonify==True:
            audioplayer(samples, fs, name)
            name_and_error = name + " with " + error_type
            audioplayer(errored_samples, fs,name_and_error)
        if sonify==False and index != 0:
        ##If we are at a 1% interval of the list
            if index % (len(id_list)//100) == 0:
                print((index/len(id_list))*100, "% complete")
        if output==True:
            errored_path=name.replace("Medley-solos-DB/","errored_audio/")
            sf.write(errored_path, errored_samples, fs)


# In[14]:


network = neural_network.MLPClassifier(hidden_layer_sizes=(20,3, 7), max_iter=100);
network.fit(train_set, train_errors);
pred = network.predict(test_set)
print("\r\nNeural Network Classification");
print(f1_score(test_errors,pred, average='weighted'))
print(confusion_matrix(test_errors,pred));


# In[15]:


# Use the ML system to choose the error for the test data
#process(pred,test_labels,tracks_test, sonify=True,output=False)


# In[16]:


# Use the annotated labels to choose the error for the training data
#process(train_errors,train_labels,tracks_train, sonify=False,output=True)


# In[17]:


#Use the annotated labels to choose the error for the validation data
#process(validate_errors,validate_labels,tracks_validate, sonify=False,output=True)


# In[18]:


#%rm ./errored_audio/*


# In[19]:


# Create Errored Data
path='errored_audio/'

# Change this to True if you want to load prevously-computed features
load_saved_tests = True

if not load_saved_tests:
    test_set_error, tel, tee = create_feature_set(tracks_test,path)
    print("Test Set: " + str(test_set_error.shape))
    train_set_error, trl, tre = create_feature_set(tracks_train,path)
    print("Training Set: " + str(train_set_error.shape))
    validate_set_error, val, vae = create_feature_set(tracks_validate,path)
    print("Validation Set: " + str(validate_set_error.shape))
    np.savetxt('test_set_error.csv', test_set_error, delimiter=',')
    np.savetxt('train_set_error.csv', train_set_error, delimiter=',')
    np.savetxt('validate_set_error.csv', validate_set_error, delimiter=',')
else:
    test_set_error = np.loadtxt('test_set_error.csv',delimiter=',')
    train_set_error = np.loadtxt('train_set_error.csv',delimiter=',')
    validate_set_error = np.loadtxt('validate_set_error.csv',delimiter=',')
    print("Test Set: " + str(test_set_error.shape))
    print("Training Set: " + str(train_set_error.shape))
    print("Validation Set: " + str(validate_set_error.shape))


# In[20]:


train_error_mean, train_error_std_dev = get_stats(train_set_error);
train_set_error = (train_set - train_error_mean.T)/train_error_std_dev.T
test_set_error = (test_set - train_error_mean.T)/train_error_std_dev.T
validate_set_error = (validate_set - train_error_mean.T)/train_error_std_dev.T


# In[21]:


def mix_data(set1,set2):
    if len(set1) != len(set2):
        return False
    for index in range(len(set1)):
        r = np.random.random_sample();
        if r > 0.5:
    #if index % 2 == 0:
          set1[index]=set2[index]
    return set1;


# In[22]:


#Clean Train - Clean Test
network1 = neural_network.MLPClassifier(hidden_layer_sizes=(50,100, 50), max_iter=200);
network1.fit(train_set, train_labels);
pred1 = network1.predict(test_set)
fscore1 = f1_score(test_labels,pred1, average='weighted')
print("\r\nClean Train - Clean Test");
print(fscore1)
print(confusion_matrix(test_labels,pred1));


# In[23]:


#Clean Train - Mixed Test
network2 = neural_network.MLPClassifier(hidden_layer_sizes=(50,100, 50), max_iter=200);
network2.fit(train_set, train_labels);
pred2 = network2.predict(mix_data(test_set,test_set_error))
fscore2 = f1_score(test_labels,pred2, average='weighted')
print("\r\nClean Train - Mixed Test");
print(fscore2)
print(confusion_matrix(test_labels,pred2));


# In[24]:


#Mixed Train - Clean Test
network3 = neural_network.MLPClassifier(hidden_layer_sizes=(50,100, 50), max_iter=200);
network3.fit(mix_data(train_set,train_set_error), train_labels);
pred3 = network3.predict(test_set)
fscore3 = f1_score(test_labels,pred3, average='weighted')
print("\r\nMixed Train - Clean Test");
print(fscore3)
print(confusion_matrix(test_labels,pred3));


# In[25]:


#Mixed Train - Mixed Test
network4 = neural_network.MLPClassifier(hidden_layer_sizes=(50,100, 50), max_iter=200);
network4.fit(mix_data(train_set,train_set_error), train_labels);
pred4 = network4.predict(mix_data(test_set,test_set_error))
fscore4=f1_score(test_labels,pred4, average='weighted')
print("\r\nMixed Train - Mixed Test");
print(fscore4)
print(confusion_matrix(test_labels,pred4));


# In[26]:


clean_avg=(fscore1+fscore2)/2
mixed_avg=(fscore3+fscore4)/2
clean_clean_vs_clean_mixed_delta=((fscore1-fscore2)/fscore1)*100
clean_avg_vs_mixed_avg_delta=((mixed_avg-clean_avg)/clean_avg)*100
clean_clean_vs_mixed_clean_delta=((fscore1-fscore3)/fscore1)*100
print("Clean model:",clean_avg)
print("Mixed model:",mixed_avg)
print("Mixed Trained Model AVG improvement",clean_avg_vs_mixed_avg_delta)
print("Deterioration of Clean trained model when given mixed data",clean_clean_vs_clean_mixed_delta)
print("Clean trained model improvement over dirty trained model when given clean data",clean_clean_vs_mixed_clean_delta)


# In[27]:


#Augmented vs Mixed Data
large_set=np.append(test_set,test_set_error,axis=0)
large_labels=np.append(test_labels,test_labels,axis=0)


# In[28]:


#Clean Train - Clean Test
network5 = neural_network.MLPClassifier(hidden_layer_sizes=(50,100, 50), max_iter=200);
network5.fit(train_set, train_labels);
pred5 = network5.predict(test_set)
fscore5 = f1_score(test_labels,pred5, average='weighted')
print("\r\nClean Train - Clean Test");
print(fscore5)
print(confusion_matrix(test_labels,pred5));


# In[29]:


#Clean Train - Augmented Test
network6 = neural_network.MLPClassifier(hidden_layer_sizes=(50,100, 50), max_iter=200);
network6.fit(train_set, train_labels);
pred6 = network6.predict(large_set)
fscore6 = f1_score(large_labels,pred6, average='weighted')
print("\r\nClean Train - Augmented Test");
print(fscore6)
print(confusion_matrix(large_labels,pred6));


# In[30]:


#Clean Train - Dirty Test
network9 = neural_network.MLPClassifier(hidden_layer_sizes=(50,100, 50), max_iter=200);
network9.fit(train_set, train_labels);
pred9 = network9.predict(test_set_error)
fscore9 = f1_score(test_errors,pred9, average='weighted')
print("\r\nClean Train - Dirty Test");
print(fscore9)
print(confusion_matrix(test_errors,pred9));


# In[31]:


#Augmented Train - Clean Test
network7 = neural_network.MLPClassifier(hidden_layer_sizes=(50,100, 50), max_iter=200);
network7.fit(np.append(train_set,train_set_error,axis=0), np.append(train_labels,train_labels));
pred7 = network7.predict(test_set)
fscore7 = f1_score(test_labels,pred7, average='weighted')
print("\r\nAugmented Train - Clean Test");
print(fscore7)
print(confusion_matrix(test_labels,pred7));


# In[32]:


#Augmented Train - Augmented Test
network8 = neural_network.MLPClassifier(hidden_layer_sizes=(50,100, 50), max_iter=200);
network8.fit(np.append(train_set,train_set_error,axis=0), np.append(train_labels,train_labels));
pred8 = network8.predict(large_set)
fscore8=f1_score(large_labels,pred8, average='weighted')
print("\r\nAugmented Train - Augmented Test");
print(fscore8)
print(confusion_matrix(large_labels,pred8));


# In[33]:


#Augmented Train - Dirty Test
network10 = neural_network.MLPClassifier(hidden_layer_sizes=(50,100, 50), max_iter=200);
network10.fit(np.append(train_set,train_set_error,axis=0), np.append(train_labels,train_labels));
pred10 = network10.predict(large_set)
fscore10=f1_score(large_labels,pred10, average='weighted')
print("\r\nAugmented Train - Dirty Test");
print(fscore10)
print(confusion_matrix(large_labels,pred10));


# In[35]:


clean_avg=(fscore5+fscore6+fscore9)/3
mixed_avg=(fscore7+fscore8+fscore10)/3
clean_clean_vs_clean_dirty_delta=((fscore5-fscore9)/fscore5)*100
clean_avg_vs_mixed_avg_delta=((mixed_avg-clean_avg)/clean_avg)*100
clean_clean_vs_mixed_clean_delta=((fscore5-fscore7)/fscore5)*100
print("Clean model:",clean_avg)
print("Mixed model:",mixed_avg)
print("Mixed Trained Model AVG improvement",clean_avg_vs_mixed_avg_delta)
print("Deterioration of Clean trained model when given mixed data",clean_clean_vs_clean_mixed_delta)
print("Clean Model vs Augmented trained model (Clean Test)",clean_clean_vs_mixed_clean_delta)

