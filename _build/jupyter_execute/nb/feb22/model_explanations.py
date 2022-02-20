#!/usr/bin/env python
# coding: utf-8

# # model explanations

# Here we explain the linear model's predictions using Shap explanation methods.

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


# ## load the data and the trained model

# In[3]:


# data
test_file = f"{proj_dir}/data/emoprox2/test_dataframe.pkl"
test_df = pd.read_pickle(test_file)
display(test_df)

X, y = prepare_data_for_model(df=test_df)

# model
model_file = f"{results_dir}/models/linear_model"
linear_model = tf.keras.models.load_model(model_file)

linear_model.summary()


# ## explanations

# ### data for explaining the model

# In[4]:


# x, y = prepare_data_for_model(test_df[:1], shuffle=False)
x = np.concatenate(test_df[:1]['X'].tolist())
y = np.concatenate(test_df[:1]['y'].tolist())


# ### shap: GradientExplainer

# In[5]:


# select a set of background examples to take an expectation over
background = shap.utils.sample(X, nsamples=50)
# background.shape

gradient_explainer = shap.GradientExplainer(linear_model, background)


# In[6]:


shap_values_gradient = gradient_explainer.shap_values(x)


# In[7]:


shap.summary_plot(shap_values_gradient[0], x, max_display=x.shape[-1])


# ### shap: KernelExplainer

# In[ ]:


# select a set of background examples to take an expectation over
# 50 'typical' feature values
background = shap.utils.sample(X, nsamples=50)
# background.shape

kernel_explainer = shap.KernelExplainer(linear_model, background)

# 500 perturbation samples over the typical values to compute shap values for the given dataset.
shap_values_kernel = kernel_explainer.shap_values(x, nsamples=500)


# In[9]:


shap.summary_plot(shap_values_kernel[0], x, max_display=x.shape[-1])

shap.force_plot(kernel_explainer.expected_value[0], shap_values_kernel[0][0])fig, ax = plt.subplots(figsize=(25, 5), dpi=150)
ax.plot(y)
ax.set_title(f"proximity values")
plt.grid(True)