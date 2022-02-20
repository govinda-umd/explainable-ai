#!/usr/bin/env python
# coding: utf-8

# # January 27-30, 22

# In[1]:


import os
import sys
from os.path import join as pjoin


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import pickle 

import shap

# main dirs
proj_dir = pjoin(os.environ['HOME'], 'explainable-ai')

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


# ## model

# In[2]:


model = Linear_Model()

default_slice = lambda x, start, end : x[start : end, ...]

linear_regression = base_model(task_type="regression", 
                               model=model, 
                               loss_object=tf.keras.losses.MeanSquaredError(), 
                               L1_scale=0.0, 
                               L2_scale=0.0,
                               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), 
                               eval_metric=tfa.metrics.RSquare(),
                               eval_metric_name="% var explained",
                               batch_size=32, 
                               slice_input=default_slice)


# ## data

# In[3]:


# get data
with open(pjoin(proj_dir, 'data/emoprox2', 'train_test_arrays.pkl'), 'rb') as f:
    data_dict = pickle.load(f)

# converting to tf tensors
data_dict['train'] = to_tensor(data_dict['train'])
data_dict['test'] = to_tensor(data_dict['test'])

# get inputs, targets and masks
train_X = data_dict['train'][0]
train_y = data_dict['train'][1]
train_mask = data_dict['train'][2]

test_X = data_dict['test'][0]
test_y = data_dict['test'][1]
test_mask = data_dict['test'][2]

# mask the tensors
train_X = train_X * tf.expand_dims(tf.cast(train_mask, 'float32'), -1)
train_y = train_y * tf.cast(train_mask, 'float32')

test_X = test_X * tf.expand_dims(tf.cast(test_mask, 'float32'), -1)
test_y = test_y * tf.cast(test_mask, 'float32')


# In[4]:


tf.expand_dims(tf.cast(train_mask, 'float32'), -1).shape

print('mask ----')
mask = np.array([[True, False], [False, True]])
mask = tf.convert_to_tensor(mask)
print(mask.shape)
print(mask)

print('####')

print('X ----')
X = np.array([[[1.0, 2.0, 1.0], [2.0, 1.0, 2.0]], 
              [[3.0, 4.0, 3.0], [4.0, 3.0, 4.0]]])
X = X * tf.expand_dims(tf.cast(mask, 'float32'), -1)
print(X.shape)
print(X)

print('####')

print('y ----')
y = np.array([
    [[1.0], [1.0]], 
    [[2.0], [2.0]]
])
y = y * tf.expand_dims(tf.cast(mask, 'float32'), -1)
print(y.shape)
print(y)

print('####')
# ## train the model
y_pred = linear_regression.model(X)

# print(linear_regression.evaluate(X, y, False))
# print(linear_regression.evaluate(X, y, True))
# print(linear_regression.evaluate(X, y, False))
# print(linear_regression.evaluate(X, y, True))

# linear_regression.fit(train_X=X, train_Y=y, val_X=X, val_Y=y, num_epochs=3)y_pred = linear_regression.model(train_X)

print( linear_regression.evaluate(train_X, train_y, False) )
print( linear_regression.evaluate(train_X, train_y, False) )
print( linear_regression.evaluate(train_X, train_y, True) )
print( linear_regression.evaluate(train_X, train_y, False) )
# In[9]:


results = linear_regression.fit(train_X=train_X, 
                                train_Y=train_y, 
                                val_X=train_X, 
                                val_Y=train_y, 
                                num_epochs=15)


# ## Shapley values
# select a set of background examples to take an expectation over
s = train_X.shape
X = tf.reshape(train_X, shape=(s[0]*s[1], s[2])).numpy()
X_background = shap.utils.sample(X, 100)explainer = shap.KernelExplainer(model=linear_regression.model, 
                                 data=X_background)
shap_values = explainer.shap_values(test_X[0, 0:1, :].numpy())shap_values.shape