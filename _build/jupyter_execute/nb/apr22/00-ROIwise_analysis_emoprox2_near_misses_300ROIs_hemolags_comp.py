#!/usr/bin/env python
# coding: utf-8

# # April 11, 2022: Schaefer parcellation (300), near misses dataset, hemodynamic lags comparison
# Compare performance of models on near miss segments with different hemodynamic shifts.

# In[1]:


import os
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa
import pickle, time, random
import neural_structured_learning as nsl
from tqdm import tqdm
import json
from itertools import combinations

# explanation tools
import shap

# plotting
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

import hypernetx as hnx
from networkx import fruchterman_reingold_layout as layout

# nilearn
from nilearn import image
from nilearn import masking
from nilearn import plotting

# main dirs
proj_dir = pjoin(os.environ['HOME'], 'explainable-ai')
results_dir = f"{proj_dir}/results"
month_dir = f"{proj_dir}/nb/mar22"

# folders
sys.path.insert(0, proj_dir)
from helpers.dataset_utils import *
from helpers.base_model import *
from helpers.model_definitions import *

# select the GPU to be used
gpus = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

# print the JS visualization code to the notebook
shap.initjs()


# ## emoprox2 full datasets and models

# In[2]:


hemo_lags = [0, 2, 4, 6, 8]
datas = []
accuracies = []


# In[3]:


def get_data_samples(data, subject_list):
    
    X = []; y = []
    target = np.expand_dims(np.array([0,0,0,0,0,0,0,1,1,1,1,1]),axis=0).astype(np.float64)
    
    for subject in subject_list:
        
        num_samples = data[subject].shape[0]
        X.append(data[subject])
        y.append(np.repeat(target, num_samples, axis=0))

    return np.vstack(X), np.vstack(y)


# In[4]:


def model(hemo_lag):
    model_file = f"{results_dir}/emoprox_full_data/models/GRU_classifier_nearmiss_hemolag{hemo_lag}"
    history_file = f"{results_dir}/emoprox_full_data/models/GRU_classifier_nearmiss_hemolag{hemo_lag}_history"
    if os.path.exists(model_file):
        # load the model
        model = tf.keras.models.load_model(model_file)
        history = json.load(open(f"{history_file}", 'r'))
    else:
        # build, train, and save the model
        '''
        build model
        '''
        model = get_GRU_classifier_model(
            X_train, 
            args, 
            regularizer)

        '''
        train model
        '''
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        then = time.time()
        history = model.fit(
            x=X_train, 
            y=y_train,
            batch_size=args.batch_size, 
            epochs=args.num_epochs, 
            verbose=1,
            callbacks=tf.keras.callbacks.EarlyStopping(patience=5),
            validation_split=args.validation_split, 
            shuffle=True)
        print('--- train time =  %0.4f seconds ---' %(time.time() - then))

        '''
        save model
        '''
        model.save(model_file)
        history = history.history
        json.dump(history, open(f"{history_file}", 'w'))
        
    # evaluate the model
    eval_hist = model.evaluate(X_test, y_test)
    return eval_hist[1]


# In[5]:


class ARGS(): pass
args = ARGS()

args.SEED = 74

# model
args.num_units = 32
args.num_classes = 2 # for binary classification
args.l2 = 1e-2
args.dropout = 0.8
args.learning_rate = 4e-4

args.num_epochs = 100
args.validation_split = 0.2
args.batch_size = 64

args.temp = 20


# In[6]:


regularizer = tf.keras.regularizers.l2(l2=args.l2) 
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)


# In[7]:


for hemo_lag in hemo_lags:
    raw_data_file = (f"/home/joyneelm/approach-retreat/near_miss_analysis/classification_data"
                    f"/Schaefer2018_roi300_net17_122subjs_nearmiss_segments_withoutshock_hemodynamic_lag{hemo_lag}.pkl")
    with open(raw_data_file, 'rb') as f:
        data = pickle.load(f)
        datas.append(data)
    
    # data
    args.num_subjects = len(data.keys())
    args.num_train = args.num_subjects // 2
    args.num_test = args.num_subjects - args.num_train
    subject_list = list(data.keys())
    random.Random(args.SEED).shuffle(subject_list)

    train_list = subject_list[:args.num_train]
    test_list = subject_list[args.num_train:]

    (X_train, y_train) = get_data_samples(data, train_list)
    (X_test, y_test) = get_data_samples(data, test_list)
    tf.random.set_seed(args.SEED)

    accuracies.append(model(hemo_lag))


# In[8]:


accuracies


# In[9]:


fig, ax = plt.subplots(
    nrows=1, 
    ncols=1, 
    figsize=(11, 5), 
    sharex=True, 
    sharey=True, 
    dpi=150)
ax.plot(accuracies, color='blue', linestyle='-', marker='o')
ax.set_xlabel('hemo lags')
ax.set_ylabel('accuracy')
ax.set_xticks(np.arange(len(hemo_lags)))
ax.set_xticklabels(hemo_lags)
ax.set_ylim(0.85, 1.0)
ax.grid(True)
plt.show()

