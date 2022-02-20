#!/usr/bin/env python
# coding: utf-8

# # February 1-2, 10-14, 22

# In[1]:


import os
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa
# import pickle 
from tqdm import tqdm

# explanation tools
import shap

# plotting
import matplotlib.pyplot as plt
plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 9
plt.rcParams["errorbar.capsize"] = 0.5

# nilearn
from nilearn import image
from nilearn import masking
from nilearn import plotting

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

# print the JS visualization code to the notebook
shap.initjs()


# In[2]:


tf.__version__


# ## data

# In[3]:


file_name = f"{proj_dir}/data/emoprox2/dataframe.pkl"
data_df = pd.read_pickle(file_name)
data_df


# In[4]:


# split subjects into train and test partitions
subjs = pd.unique(data_df['subj'])
train_subjs, test_subjs = split_subjs(subjs, 0.9)

train_df = get_Xy(data_df, train_subjs)
test_df = get_Xy(data_df, test_subjs)
test_df


# In[5]:


# apply censor mask
train_df = apply_mask_Xy(train_df)
test_df = apply_mask_Xy(test_df)


# In[6]:


# prepare input and target for model
X_train, y_train = prepare_data_for_model(df=train_df)
X_test, y_test = prepare_data_for_model(df=test_df)


# In[7]:


print(X_train.shape, y_train.shape)


# In[8]:


print(X_test.shape, y_test.shape)


# ## model

# In[9]:


'''
linear regression model
'''
# input layer: TRY TO USE EMBEDDING LAYER AFTER CLEARING FIRST RUN
inputs = tf.keras.Input(shape=(X_train.shape[-1]))

# preprocessing layer
# normalizer = tf.keras.layers.Normalization(axis=-1)
# normalizer.adapt(np.array(X_train))
# linear_model.add(normalizer)
# x = normalizer(inputs)

# linear layer
outputs = tf.keras.layers.Dense(units=1)(inputs)

# model 
linear_model = tf.keras.Model(inputs=inputs, outputs=outputs)

linear_model.summary()


# In[10]:


'''
compiling and training
'''
linear_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.1),
                     loss=tf.keras.losses.MeanSquaredError())
# metrics=tfa.metrics.RSquare()

linear_model.fit(x=X_train, 
                 y=y_train, 
                 batch_size=32, 
                 epochs=10, 
                 verbose='auto',
                 callbacks=tf.keras.callbacks.EarlyStopping(patience=5),
                 validation_split=0.2, 
                 shuffle=True)


# In[11]:


y_pred = linear_model.predict(X_test)
print(y_pred.shape)
print(y_pred.flatten().shape)


# ## shap

# ### KernelExplainer: works!
# SHAP expects model functions to take a 2D numpy array as input
def f(X):
    return linear_model.predict(X).flatten()Here we use a selection of 50 samples from the dataset to represent “typical” feature values, and then use 500 perterbation samples to estimate the SHAP values for a given prediction. Note that this requires 500 * 50 evaluations of the model.# select a set of background examples to take an expectation over
background = shap.utils.sample(X_train, nsamples=50)
# background.shape

# explain prediction of the model 
# 50 'typical' feature values
explainer = shap.KernelExplainer(f, background)
# 500 perturbation samples over the typical values to compute shap values for the given dataset.
shap_values = explainer.shap_values(X_test[4:5], nsamples=500) shap.force_plot(explainer.expected_value, shap_values, X_test[4:5])shap_values_all_test = explainer.shap_values(X_test[:50], nsamples=500, verbose=0)shap.force_plot(explainer.expected_value, shap_values_all_test)
# ### Explainer - Partition explainer: works!
# create an explainer with model and a masker
# masker masks out tabular features by integrating over the given background dataset
explainer = shap.Explainer(linear_model.predict, masker=shap.maskers.Partition(background))

# here we explain two images using 500 evaluations of the underlying model to estimate the SHAP values
shap_values = explainer(X_test[:25], max_evals=171, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])
np.squeeze(shap_values[:5].base_values)
# ### DeepExplainer: works!

# In[12]:


# select a set of background examples to take an expectation over
background = shap.utils.sample(X_train, nsamples=50)
# background.shape

deep_explainer = shap.DeepExplainer(linear_model, background)
shap_values = deep_explainer.shap_values(X_test[:])


# In[ ]:





# ### GradientExplainer: works!

# In[13]:


# select a set of background examples to take an expectation over
background = shap.utils.sample(X_train, nsamples=50)
# background.shape

explainer = shap.GradientExplainer(linear_model, background)


# In[14]:


shap_values = explainer.shap_values(X_test[:])


# In[15]:


shap_values[0].shape


# ## visualizing on the brain

# In[16]:


def plot_roi_vec_on_niimg(roi_data, mask):
    # create an empty stat img
    stat_img_all_rois = image.new_img_like(ref_niimg=mask, 
                                           data=np.zeros_like(mask.get_fdata(), 
                                                              dtype=np.float32), 
                                           copy_header=True)
    
    # unmask roi value on all voxels of the roi
    for idx_roi in tqdm(np.arange(roi_data.shape[-1])):
        mask_roi = image.math_img(f"img=={idx_roi+1}", img=mask)
        num_voxels = np.where(mask_roi.get_fdata())[0].shape[0]
        vox_data = roi_data[idx_roi] * np.ones(shape=(num_voxels,))
        stat_img = masking.unmask(vox_data, mask_img=mask_roi)
        stat_img_all_rois = image.math_img(f"img_all+img_roi", 
                                           img_all=stat_img_all_rois, 
                                           img_roi=stat_img)
        
    return stat_img_all_rois


# In[17]:


mask_file = f"/home/govindas/parcellations/templates/MAX_ROIs_final_gm_85.nii.gz"
# print(mask_file)
mask = image.load_img(mask_file)


# In[18]:


idx_time = 1
roi_data = shap_values[0][idx_time]
stat_img_all_rois = plot_roi_vec_on_niimg(roi_data=roi_data, 
                                          mask=mask)


# In[19]:


plotting.plot_stat_map(stat_img_all_rois, colorbar=True)

