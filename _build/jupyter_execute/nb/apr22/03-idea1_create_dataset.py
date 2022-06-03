#!/usr/bin/env python
# coding: utf-8

# # April 25-30, 2022: Idea1, don't-care time segments: create dataset

# In[1]:


import os
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
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
month_dir = f"{proj_dir}/nb/apr22"

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


def get_subj_timing_dfs(base_timing_dfs, subj_timings, run_list):
    '''
    base_timing_dfs : per run, per block
    block_timings: per run
    shock_timings: per run, split per block
    '''
    subj_timing_dfs = copy.deepcopy(base_timing_dfs)
    for run, block in list(product(run_list, range(2))):
        block_base_timing_df = base_timing_dfs[run][block].copy()
        block_timings, shock_timings, run_duration = subj_timings[run]

        # separate appr and retr stimuli
        block_base_timing_df['appr'] = (block_base_timing_df['direction'] == 1.0).values.astype(float)
        block_base_timing_df['retr'] = (block_base_timing_df['direction'] == -1.0).values.astype(float)

        # time in seconds
        block_start, block_stop = block_timings[block]
        block_base_timing_df['time'] = np.linspace(block_start, block_stop, num=len(block_base_timing_df))

        # times of electrical stimulation
        block_base_timing_df['shock_time'] = np.zeros_like(block_base_timing_df['time'])
        for idx_shock, shock in enumerate(shock_timings[block]):
            idx_shock_start = abs(block_base_timing_df['time'] - shock[0]).idxmin()
            idx_shock_stop = abs(block_base_timing_df['time'] - shock[1]).idxmin() 
            block_base_timing_df['shock_time'][idx_shock_start: idx_shock_stop] = 1.0
        
        # store properly
        subj_timing_dfs[run][block] = block_base_timing_df  
    return subj_timing_dfs


# In[3]:


a = [0,2,4]
b = [0,1]
list(product(enumerate(a), b))


# In[4]:


# convolve with hrf
BASECOLS = ['proximity', 'direction', 'appr', 'retr', 'shock_time']
MAX_TIME = 500 # in seconds

def h(t, p=8.6, q=0.547, tau=5.0):
    # return 1.0 if ((t >= 0.0) and (t <= tau)) else 0.0 
    return np.power((t/(p*q)), p) * np.exp(p-t/q)

def take_samples(signal, times):
    signal_discrete = np.zeros_like(times)
    for idx_time, time in enumerate(times):
        signal_discrete[idx_time] = signal(time)
    return signal_discrete

def convolve_hrf(x, hrf, delta_t):
    y = np.convolve(x, hrf, 'full') * delta_t 
    # this is approximation of the integration done in convolution of two continuous signals. 
    # since those two signals were sampled, we need to multiply the product 
    # of the two signals with `delta_t` to approximate the integration.

    # but here for visualizing/ using regressors similar to how afni does, 
    # we are normalizing the convolved response
    y /= np.max(np.abs(y))
    return y #y[:x.shape[0]]

def get_convolved_regressors(subj_timing_dfs, run_list):
    for run, block in list(product(run_list, range(2))): 
        df = subj_timing_dfs[run][block]

        # sample the hrf kernel 
        # as block0 has time starting from 0 seconds, use times from there.
        t = subj_timing_dfs[run][0]['time'][0:500]
        delta_t = t[1] - t[0]
        hrf = take_samples(h, t)

        # create a new longer df to include convolved signals 
        df_conv = pd.DataFrame(
            np.nan, 
            index=np.arange(len(df)+hrf.shape[0]-1), 
            columns=df.columns)
        df_conv.iloc[:len(df)] = df.iloc[:]

        # add times in seconds
        times = df['time']
        dt = times[1] - times[0]
        t_stop = MAX_TIME #times.iloc[0] + (len(df_conv)) * dt
        df_conv['time'] = np.arange(start=times.iloc[0], stop=t_stop, step=dt)[:len(df_conv)]

        # convolve base signals/ columns of df
        for COL in BASECOLS:
            x = df[COL]
            y = convolve_hrf(x, hrf, delta_t)
            df_conv[f"{COL}_conv"] = y
        
        subj_timing_dfs[run][block] = df_conv
    
    return subj_timing_dfs


# In[5]:


# times closest to each TR: 
# indexes where data will be sampled at every TR
# idx == -1 implies that TR does not lie inside either block.
tr = 1.25
def return_closest_time(df, fps=1/30):
    TR_list = np.arange(0,360*tr,tr)
    diff_thresh = fps
    
    idx_list = -1 * np.ones(len(TR_list), dtype=np.int)

    for i, TR in enumerate(TR_list):
        diff = np.abs(df.time.values - TR)
        if np.min(diff) <= diff_thresh:
            idx_list[i] = np.argmin(diff)
    
    return idx_list


# In[6]:


def get_subj_tr_dfs(subj, run_list):
    subj_timings = [None for _ in np.arange(6)]
    for idx_run, run in enumerate(run_list):
        subj_timings[run] = stimulus_utils.subj_timing(subj, run)
    # block_timings, shock_timings, run_duration = \
    # stimulus_utils.subj_timing(subj, run)

    # align the base stimuli to the scanning-times of the subject
    subj_timing_dfs = get_subj_timing_dfs(base_timing_dfs, subj_timings, run_list)

    # convolve base stimuli with hrf kernel
    subj_timing_dfs = get_convolved_regressors(subj_timing_dfs, run_list)

    # resample the block-wise dataframes to TR resolution
    idx_lists = [None for _ in np.arange(6)]
    for run in run_list:
        idx_lists[run] = return_closest_time(
            pd.concat([subj_timing_dfs[run][0], subj_timing_dfs[run][1]]))

    subj_tr_dfs = [None for _ in np.arange(6)] 
    # single df per run, and time will be indexed by TR

    for run in run_list:
        idx_list = idx_lists[run]
        df = pd.concat([subj_timing_dfs[run][0], subj_timing_dfs[run][1]])
        df_tr = pd.DataFrame(
            np.nan, 
            index=np.arange(len(idx_list)), 
            columns=df.columns)

        blocks = dataset_utils.contiguous_regions(idx_list!=-1)
        for idx_block in [0, 1]:
            block = blocks[idx_block]
            df_tr.iloc[block[0]:block[1]] = df.iloc[idx_list[block[0]:block[1]]]

        tr_dfs[run] = df_tr
    
    return subj_tr_dfs


# In[7]:


base_timing_dfs = [[stimulus_utils.get_base_stimulus(run, block) for block in range(2)] for run in range(6)]


# In[8]:


raw_path = '/home/joyneelm/approach-retreat/data/raw'
yoked = pd.read_excel(f"{raw_path}/CON_yoked_table.xlsx", engine='openpyxl')
yoked = yoked.query('use == 1')


# In[ ]:


tr_dfs = {}
for _, row in yoked.iterrows():
    for kind in ['control', 'uncontrol']:
        subj = row[kind].replace('CON', '')
        run_list = np.arange(6)[row.loc['run0':'run5'].astype(bool)]

        print(f"{subj} : {run_list}")

        tr_dfs[subj] = get_subj_tr_dfs(subj, run_list)


# In[13]:


run_list


# In[12]:


tr_dfs['045']


# #### plots of all stimuli/regressors

# In[36]:


nrows, ncols = 12, 1
subj = '045'
l = 180 # half length of a run
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5*nrows), dpi=150)
for run, block in list(product(run_list, range(2))):
    idx = 2*run + block
    ax = axs[idx]
    ax.plot(tr_dfs[subj][run]['appr'][block*l : (block+1)*l], 
            color='peru', linestyle='-.', label='appr')
    ax.plot(
        tr_dfs[subj][run]['appr_conv'][block*l : (block+1)*l], 
        color='salmon', linestyle='-', label='appr_conv')
    ax.plot(
        tr_dfs[subj][run]['retr'][block*l : (block+1)*l], 
        color='slateblue', linestyle='-.', label='retr')
    ax.plot(
        tr_dfs[subj][run]['retr_conv'][block*l : (block+1)*l], 
        color='cornflowerblue', linestyle='-', label='retr_conv')
    ax.plot(
        tr_dfs[subj][run]['proximity'][block*l : (block+1)*l],
        color='mediumseagreen', linestyle='-.', label='prox')
    ax.plot(
        tr_dfs[subj][run]['proximity_conv'][block*l : (block+1)*l],
        color='green', linestyle='-', label='prox_conv')
    ax.plot(
        tr_dfs[subj][run]['shock_time'][block*l : (block+1)*l],
        color='grey', linestyle='-.', label='shock')
    ax.plot(
        tr_dfs[subj][run]['shock_time_conv'][block*l : (block+1)*l],
        color='black', linestyle='-', label='shock_conv')

    ax.set_title(f"run{run}_block{block}")
    ax.legend()
    ax.grid(True)

