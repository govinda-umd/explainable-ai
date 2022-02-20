#!/usr/bin/env python
# coding: utf-8

# # get dataset: emoproxII

# In[1]:


import os
import sys
from os.path import join as pjoin


import numpy as np 
import pandas as pd 
from glob import glob
import pickle 

# main dirs
proj_dir = pjoin(os.environ['HOME'], 'explainable-ai')

# folders
sys.path.insert(0, proj_dir)
from helpers.dataset_utils import *
from helpers.base_model import *
from helpers.model_definitions import *


# In[2]:


main_path = f"/home/joyneelm/approach-retreat/data"
time_series_path = join(main_path, "interim/CON{subj}/CON{subj}_MAX_rois_meanTS.1D")
target_path = join(main_path, "raw/CON{subj}/regs_fancy/CON{subj}_all_regs.txt")


# ## organize data per run in a df 

# In[3]:


subj_folder_list = glob(f"{main_path}/raw/*", recursive=False)
num_TRs = 360 # per run

data_df = get_data_df(subj_folder_list, time_series_path, target_path, num_TRs)
data_df


# In[4]:


# save the dataframe
file_name = f"{proj_dir}/data/emoprox2/dataframe.pkl"
data_df.to_pickle(file_name)

## create input and target vectors# split subjects into train and test partitions
subjs = pd.unique(data_df['subj'])
train_subjs, test_subjs = split_subjs(subjs, 0.9)

# create X and y
train_arrays = get_Xy(data_df, train_subjs) # (X_train, y_train, mask_train)
test_arrays = get_Xy(data_df, test_subjs) # (X_test, y_test, mask_test)# save these arrays
with open(pjoin(proj_dir, 'data/emoprox2', 'train_test_arrays.pkl'), 'wb') as f:
    pickle.dump({'train':train_arrays, 'test':test_arrays}, f)