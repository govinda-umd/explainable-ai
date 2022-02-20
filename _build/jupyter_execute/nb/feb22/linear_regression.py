#!/usr/bin/env python
# coding: utf-8

# # February 15, 2022

# Here, we explain the predictions of a simple linear regression model trained on emoprox2 dataset.

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

# plotting/visualizations
import sklearn as skl

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
results_dir = f"{proj_dir}/results"

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


train_file = f"{proj_dir}/data/emoprox2/train_dataframe.pkl"
test_file = f"{proj_dir}/data/emoprox2/test_dataframe.pkl"

train_df = pd.read_pickle(train_file)
test_df = pd.read_pickle(test_file)


# In[4]:


# prepare input and target for model
X_train, y_train = prepare_data_for_model(df=train_df)
X_test, y_test = prepare_data_for_model(df=test_df)


# In[5]:


print(X_train.shape, y_train.shape)


# In[6]:


print(X_test.shape, y_test.shape)


# ## model

# In[7]:


'''
linear regression model
'''
linear_model = get_linear_model(X_train)

print(linear_model.summary())

'''
training 
'''
linear_model.fit(x=X_train, 
                 y=y_train, 
                 batch_size=32, 
                 epochs=50, 
                 verbose='auto',
                 callbacks=tf.keras.callbacks.EarlyStopping(patience=5),
                 validation_split=0.2, 
                 shuffle=True)

'''
saving
'''
linear_model.save(f"{results_dir}/models/linear_model")


# In[8]:


y_pred = linear_model.predict(X_test)
print(y_pred.shape)
print(y_pred.flatten().shape)


# ## explanations

# ### shap: GradientExplainer
# select a set of background examples to take an expectation over
background = shap.utils.sample(X_train, nsamples=50)
# background.shape

gradient_explainer = shap.GradientExplainer(linear_model, background)x, y = prepare_data_for_model(test_df, shuffle=False)
shap_values = gradient_explainer.shap_values(x)def plot_roi_vec_on_niimg(roi_data, mask):
    # create an empty stat img
    s = list(mask.get_fdata().shape)
    s.append(1)
    stat_img_all_rois = image.new_img_like(ref_niimg=mask, 
                                           data=np.zeros(shape=s, 
                                                         dtype=np.float32), 
                                           copy_header=True)
    
    # unmask roi value on all voxels of the roi
    num_time, num_rois = roi_data.shape
    for idx_roi in tqdm(np.arange(num_rois)):
        mask_roi = image.math_img(f"img=={idx_roi+1}", img=mask)
        num_voxels = np.where(mask_roi.get_fdata())[0].shape[0]
        vox_data = np.expand_dims(roi_data[:, idx_roi], axis=-1) *\
                   np.ones(shape=(num_time, num_voxels,))
        stat_img = masking.unmask(vox_data, mask_img=mask_roi)
        stat_img_all_rois = image.math_img(f"img_all+img_roi", 
                                           img_all=stat_img_all_rois, 
                                           img_roi=stat_img)

    return stat_img_all_roismask_file = f"/home/govindas/parcellations/templates/MAX_ROIs_final_gm_85.nii.gz"
# print(mask_file)
mask = image.load_img(mask_file)giving all timesteps at once uses all memory, it is better to give one run of a subject at a time.
# stat_img = plot_roi_vec_on_niimg(shap_values[0], mask)