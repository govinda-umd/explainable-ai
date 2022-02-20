#!/usr/bin/env python
# coding: utf-8

# # visualize the data
**Motivation**: What is this update about?  What was the motivation? Main findings? Put a brief summary here.
# Here, we 
# 1. split the whole data into training and testing sets, save them for further use, and 
# 2. visualize lower-dimensional representations of data and discover its patterns.

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


# whole data
file_name = f"{proj_dir}/data/emoprox2/dataframe.pkl"
data_df = pd.read_pickle(file_name)
data_df


# In[4]:


train_file = f"{proj_dir}/data/emoprox2/train_dataframe.pkl"
test_file = f"{proj_dir}/data/emoprox2/test_dataframe.pkl"

if not os.path.isfile(train_file) and not os.path.isfile(test_file):
    # split subjects into train and test partitions
    subjs = pd.unique(data_df['subj'])
    train_subjs, test_subjs = split_subjs(subjs, 0.9)

    train_df = get_Xy(data_df, train_subjs)
    test_df = get_Xy(data_df, test_subjs)

    # apply censor mask
    train_df = apply_mask_Xy(train_df)
    test_df = apply_mask_Xy(test_df)

    # save these sets
    train_df.to_pickle(train_file)
    test_df.to_pickle(test_file)
else:
    train_df = pd.read_pickle(train_file)
    test_df = pd.read_pickle(test_file)


# In[5]:


test_df


# In[6]:


# prepare input and target for model
X_train, y_train = prepare_data_for_model(df=train_df)
X_test, y_test = prepare_data_for_model(df=test_df)


# In[7]:


print(X_train.shape, y_train.shape)


# In[8]:


print(X_test.shape, y_test.shape)


# ## visualize data

# #### SELECT RANDOMLY A FEW DATA SAMPLES AND THEN VISUALIZE THEM. THIS WILL REDUCE THE OPERATIONAL LOAD. DO NOT GIVE ALL TRAINING SAMPLES.

# In[9]:


num_points = 1000
sample_idxs = np.random.choice(np.arange(X_train.shape[0]), size=num_points, replace=False)

X = X_train[sample_idxs]
y = y_train[sample_idxs]

print(X.shape, y.shape)


# ### pca

# In[10]:


pca = skl.decomposition.PCA(n_components=0.95)

X_pca = pca.fit_transform(X)


# In[11]:


cumsum = np.cumsum(pca.explained_variance_ratio_)
print(cumsum)

plt.plot(cumsum)

num_feats_pca = np.where(cumsum < 0.95)[0].shape[0]
print(num_feats_pca)


# In[12]:


x, y = X_pca[:, 3], X_pca[:, 5]
plt.scatter(x, y, c=y)
plt.colorbar()


# ### Isomap

# In[13]:


isomap_embedding = skl.manifold.Isomap(n_components=3, neighbors_algorithm='kd_tree', n_jobs=5)

X_isomap = isomap_embedding.fit_transform(X)


# In[14]:


cm = 1/2.54
fig = plt.figure(figsize=(15*cm, 15*cm))
ax = fig.gca(projection='3d')
x, y, z = [X_isomap[:, i] for i in [0,1,2]]
s = ax.scatter3D(x, y, z, marker='.', c=y)
fig.colorbar(s)


# ### spectral embedding

# In[15]:


spectral = skl.manifold.SpectralEmbedding(n_components=3, n_jobs=5) #n_neighbors=50

X_spectral = spectral.fit_transform(X)


# In[16]:


cm = 1/2.54
fig = plt.figure(figsize=(15*cm, 15*cm))
ax = fig.gca(projection='3d')
x, y, z = [X_spectral[:, i] for i in [0,1,2]]
s = ax.scatter3D(x, y, z, marker='.', c=y)
fig.colorbar(s)


# **Isomap and Spectral embedding**
# This is a continuous, single manifold. Data points for lower proximities are at the center of the manifold, and data points for higher proximities move away from the center in two directions. This suggests that 
# 1. data points with similar proximities are together, i.e. fmri responses capture information about proximity, 
# 2. since data points diverge into two directions, there may be *two varieties of explanations* for the same proximity value.

# ### mds: multidimensional scaling

# In[17]:


mds_embedding = skl.manifold.MDS(n_components=3, n_jobs=5)

X_mds = mds_embedding.fit_transform(X)


# In[18]:


cm = 1/2.54
fig = plt.figure(figsize=(15*cm, 15*cm))
ax = fig.gca(projection='3d')
x, y, z = [X_mds[:, i] for i in [0,1,2]]
s = ax.scatter3D(x, y, z, marker='.', c=y)
fig.colorbar(s)


# ### tsne

# In[19]:


tsne = skl.manifold.TSNE(n_components=3, 
                         learning_rate='auto', 
                         init='random', 
                         n_jobs=5)


# In[20]:


X_tsne = tsne.fit_transform(X)


# In[21]:


cm = 1/2.54
fig = plt.figure(figsize=(15*cm, 15*cm))
ax = fig.gca(projection='3d')
x, y, z = [X_tsne[:, i] for i in [0,1,2]]
s = ax.scatter3D(x, y, z, marker='.', c=y)
fig.colorbar(s)


# **MDS and tSNE**
# This single, blob-like manifold suggests that data points with similar proximities group together. 
