#!/usr/bin/env python
# coding: utf-8

# # May 20, 2022: idea1 prototyping model implementation
# check with a small dummy data whether implementations of loss and accuracy are working correctly.

# In[1]:


import os
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import tensorflow.keras.backend as K
# import tensorflow_addons as tfa
import pickle, time, random
import neural_structured_learning as nsl
from tqdm import tqdm
import json
from itertools import combinations, product
from operator import add
import copy

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
month_dir = f"{proj_dir}/nb/may22"

# folders
sys.path.insert(0, proj_dir)
import helpers.dataset_utils as dataset_utils
import helpers.base_model as base_model
import helpers.model_definitions as model_definitions
import data.emoprox2.scripts.stimulus_utils as stimulus_utils

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


# In[2]:


def get_data(data, subject_list):
    X = [] 
    y = []
    Xlen = [] 
    global MASK

    for subject in subject_list:
        X += data[subject]['X']
        y += data[subject]['y']
        Xlen += [i.shape[0] for i in data[subject]['X']]

    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X, 
        padding="post",
        dtype='float', 
        value=MASK
    )
    
    y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        y, 
        padding="post",
        dtype='float',
        value=MASK
    )
    # y_padded = tf.convert_to_tensor(y)

    return X_padded, y_padded, Xlen


# In[3]:


'''
target stimuli
'''
APPR, RETR = 1, 0
MASK = 0.5

data_file = f"{proj_dir}/data/emoprox2/idea1_data.pkl"
with open(data_file, 'rb') as f:
    data = pickle.load(f)

'''
(hyper)-parameters
'''
class ARGS(): pass
args = ARGS()

args.SEED = 74

# data args
args.num_subjects = len(data.keys())
args.num_train = args.num_subjects // 2
args.num_test = args.num_subjects - args.num_train

# model args
args.num_units = 32 #16 #32
args.num_classes = 2 # for binary classification
args.l2 = 1e-2
args.dropout = 0.0 #0.8
args.learning_rate = 4e-4

args.num_epochs = 50
args.validation_split = 0.2
args.batch_size = 64

args.return_sequences = True

'''
generate dataset for the model
'''
subject_list = list(data.keys())
random.Random(args.SEED).shuffle(subject_list)

train_list = subject_list[:args.num_train]
test_list = subject_list[args.num_train:]

(X_train, y_train, 
 len_train) = get_data(data, train_list)
(X_test, y_test, 
 len_test) = get_data(data, test_list)

print(X_train.shape, y_train.shape)


# In[4]:


class CustomSCCE(tf.keras.losses.Loss):
    '''
    Custom Sparse Categorical Crossentropy
    '''
    def __init__(self, name='custom_scce'):
        super().__init__(name=name)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, 
            reduction=tf.keras.losses.Reduction.NONE
        )
    
    def call(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != MASK, tf.float32)
        loss *= mask
        # average on non-zeros
        return tf.reduce_sum(loss) #/ tf.reduce_sum(mask)


# In[5]:


# tf.random.set_seed(args.SEED)
regularizer = tf.keras.regularizers.L2(l2=args.l2)
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

'''
build 
'''
model = model_definitions.get_GRU_classifier_model(
    X = X_train,
    args=args,
    regularizer=regularizer,
    mask_value=MASK,
    return_sequences=True,
)

print(model.summary())


# ## is loss function correct? **NO**

# In[6]:


X = X_train[2:3, ...]
y = y_train[2:3, ...]

print(X.shape, y.shape)


# In[15]:


'''
compile
'''
model.compile(
    loss=CustomSCCE(),
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

y_ = model(X)
print(np.argmax(y_, axis=-1))

print(y)

l_model = model.evaluate(X, y, batch_size=1)[0]
print(f"l_model: {l_model}")

print(f"model loss: {model.loss(y, y_)}")

l_man = CustomSCCE()(y, y_)
print(f"l_man: {l_man}")


# In[8]:


'''
compile with standard loss
'''
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

y_ = model(X)
print(np.argmax(y_, axis=-1))

print(y)

l_model = model.evaluate(X, y, batch_size=1)[0]
print(f"l_model: {l_model}")

print(f"model loss: {model.loss(y, y_)}")

l_man = CustomSCCE()(y, y_)
print(f"l_man: {l_man}")


# ## is accuracy correct? **LOOKS OK**

# In[10]:


'''
compile
'''
model.compile(
    loss=CustomSCCE(),
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

y_ = model(X)
print(y_)
print(np.argmax(y_, axis=-1))

print(y)

a_model = model.evaluate(X, y)[1]
print(a_model)

np.equal(y, np.argmax(y_, axis=-1))

