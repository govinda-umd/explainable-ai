#!/usr/bin/env python
# coding: utf-8

# # May 28, 2022: Create MAX dataset
# simple block design paradigm to study anxious apprehension by contrasting threat and touch conditions.
# threat is painful electrical stimulation and touch is a mild electric vibration.

# In[1]:


import os
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
import scipy as sp

import torch
import torch.nn as nn 
print(torch.cuda.is_available())

import pickle, time, random
# import neural_structured_learning as nsl
from tqdm import tqdm
import json
from itertools import combinations, product
from operator import add
import copy
from glob import glob

# explanation tools
import captum

# plotting
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

# nilearn
from nilearn import image
from nilearn import masking
from nilearn import plotting

# main dirs
proj_dir = pjoin(os.environ['HOME'], 'explainable-ai')
results_dir = f"{proj_dir}/results"
month_dir = f"{proj_dir}/nb/jun22"

# folders
sys.path.insert(0, proj_dir)
import helpers.dataset_utils as dataset_utils
import helpers.base_model as base_model
import helpers.model_definitions as model_definitions


# In[2]:


def get_stim_file(subj, name):
    if len(runs_to_exclude[runs_to_exclude.Subject == subj].values) == 0:
        stim_path = f"{main_data_path}/stim_times_neutral"
    else:
        stim_path = f"{main_data_path}/stim_times_neutral/{subj}"

    stim_file = []
    with open(f"{stim_path}/{name}.txt") as f:
        lines = f.read().split('\n')[:-1]
    
    for run, line in enumerate(lines):
        stim_file += [RUN_LEN*run + int(float(x) // TR) for x in line.split()]
    
    return stim_file


# In[3]:


TR = 1.25
RUN_LEN = 336
TRIAL_LEN = 14
IGNORE_IDX = -100

main_data_path = f"/home/govindas/vscode-BSWIFT-mnt/MAX"
runs_to_exclude = pd.read_csv(
    f"{main_data_path}/scripts/runs_to_exclude_neutral.txt", 
    delimiter='\t')
# runs_to_exclude

data_path = (
    f"{main_data_path}/dataset/first_level"
    f"/ROI/neutral_runs_conditionLevel_FNSandFNT/MAX_ROIs_final_gm_85"
)
subjs = os.listdir(data_path)

names = ['FNS', 'FNT']
labels = [0, 1] # safe, threat

max_data_path = f"{proj_dir}/data/max/data_df.pkl"
if not os.path.exists(max_data_path):
    subj_list, ts_list, targets_list = [], [], []
    for subj in tqdm(subjs[:]):
        subj_list.append(subj[-3:])

        # # fMRI time series
        ts = np.loadtxt(f"{data_path}/{subj}/{subj}_meanTS.1D")
        ts_list.append(ts)

        # targets
        targets = IGNORE_IDX * np.ones(ts.shape[0])
        for label, name in zip(labels, names):
            stim_file = get_stim_file(subj, name)
            for onset in stim_file:
                targets[onset:onset+TRIAL_LEN] = label
        targets_list.append(targets)

    max_data_df = pd.DataFrame(
        {
            'subj': subj_list,
            'ts': ts_list,
            'targets': targets_list
        }
    )
    with open(max_data_path, 'wb') as f:
        pickle.dump(max_data_df, f)
else:
    with open(max_data_path, 'rb') as f:
        max_data_df = pickle.load(f)


# In[4]:


max_data_df


# In[7]:


fig, ax = plt.subplots(1, 1, figsize=(20, 6), dpi=150)

ax.plot(targets)
ax.set_ylim(-1, 1)
plt.show()

