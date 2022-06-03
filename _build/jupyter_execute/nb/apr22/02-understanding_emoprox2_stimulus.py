#!/usr/bin/env python
# coding: utf-8

# # April 14,15,20,21, 2022: Understanding stimulus of emoprox2 paradigm

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
from itertools import combinations
from operator import add

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
from helpers.dataset_utils import *
from helpers.base_model import *
from helpers.model_definitions import *
from nb.apr22.ROIwise_appr_retr_timeseries_plots_helper import *

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


# ## testing convolution 

# In[2]:


# continuous time functions
def x(t, tau=10.0):
    return 1.0 if ((t >= 0.0) and (t <= tau)) else 0.0

def h(t, p=8.6, q=0.547, tau=5.0):
    # return 1.0 if ((t >= 0.0) and (t <= tau)) else 0.0 
    return np.power((t/(p*q)), p) * np.exp(p-t/q)

# sampling function
def take_samples(signal, times):
    signal_discrete = np.zeros_like(times)
    for idx_time, time in enumerate(times):
        signal_discrete[idx_time] = signal(time)
    return signal_discrete

# main code
delta = 1.25 # seconds
T = 20 # seconds
N = int((T - 0.0) / delta)

n = np.linspace(start=0.0, stop=N, num=N+1).astype(int)
t = n * delta

x_n = take_samples(x, t)
h_n = take_samples(h, t)

y_n = np.convolve(x_n, h_n)

nrows, ncols = 1, 1
figsize = (11*ncols, 5*nrows)
fig, axs = plt.subplots(
    nrows=nrows, 
    ncols=ncols, 
    figsize=figsize, 
    sharex=False, 
    sharey=False, 
    dpi=150)

ax = axs
ax.plot(t, x_n, label='input signal, $x(t)$')
ax.plot(t, h_n, label='impulse response, hrf, $h(t)$')
ax.plot(t, y_n[n], label='convolved signal, $y(t) = x(t) * h(t)$')
ax.set_xlim(0, T)
# ax.set_ylim(0.0, 4.5)
ax.set_xlabel(f"time in s")
ax.set_ylabel(f"signal value")
ax.grid(True)
ax.legend()
    


# this convolution shows that for a constant stimulus ($x(t)$), hemodynamic signal (fMRI) will peak at around 8.75TRs. But the main bulk of the signal starts from 5TRs or 6TRs.

# ## canonical fMRI responses

# In[3]:


get_ipython().run_cell_magic('bash', '', 'proj_path="/home/govindas/vscode-BSWIFT-mnt"\nmain_working_path="/home/govindas/explainable-ai"\nout_path="${main_working_path}/nb/apr22"\nsubj=\'002\'\n\n# 3dDeconvolve -x1D_stop \\\n#  -polort A \\\n#  -nodata 2160 1.25 \\\n#  -num_stimts 1 \\\n#  -stim_times_AM1 1 "${proj_path}/eCON/dataset/preproc2/CON${subj}/splitted_regs_fancy/CON${subj}_closeAllApprMR.txt" "dmBLOCK(1)" -stim_label 1 closeAppr \\\n#  -x1D ${subj}_design_mat.xmat.1D\n\n3dDeconvolve -overwrite \\\n -x1D_stop \\\n -input "${proj_path}/eCON/dataset/preproc2/CON${subj}/func/ICA_AROMA/CON${subj}_EP_TR_MNI_2mm_SI_denoised.nii.gz" \\\n -jobs 4 \\\n -mask "${proj_path}/share/templates/MNI152_T1_2mm_brain_GM_02182017.nii.gz" \\\n -polort A \\\n -local_times \\\n -concat "${proj_path}/eCON/dataset/preproc2/CON${subj}/func/runConcatInfo.1D" \\\n -censor "${proj_path}/eCON/dataset/preproc2/CON${subj}/regs_fancy/CON${subj}_censor15.txt" \\\n -num_stimts 2 \\\n -tout \\\n -stim_times_AM1 1 "${proj_path}/eCON/dataset/preproc2/CON${subj}/splitted_regs_fancy/CON${subj}_closeAllApprMR.txt" "dmBLOCK(1)"     -stim_label 1 "closeAppr" \\\n -stim_times_AM1 2 "${proj_path}/eCON/dataset/preproc2/CON${subj}/splitted_regs_fancy/CON${subj}_closeAllRetrMR.txt" "dmBLOCK(1)"     -stim_label 2 "closeRetr" \\\n -ortvec "${proj_path}/eCON/dataset/preproc2/CON${subj}/func/CON${subj}_MotionPar.txt"\'[1..6]\'  rawMotion \\\n -ortvec "${proj_path}/eCON/dataset/preproc2/CON${subj}/regs_fancy/CON${subj}_MotionParDeriv.txt"  derMotion \\\n -x1D "${out_path}/subj${subj}.xmat.1D" \\\n -x1D_uncensored "${out_path}/subj${subj}_uncens.xmat.1D" \\\n#  -bucket /data/bswift-1/Pessoa_Lab/eCON/dataset/results_ShockCensored/CON002/controllable/splitted_regs/ApprRetr_deconv/CON002_appr_bucket.nii.gz\n\nchmod +x "${main_working_path}/helpers/getVIF.py"\n# echo "${out_path}/subj${subj}.xmat.1D" | "${main_working_path}/helpers/getVIF.py" \n')


# In[4]:


subj = '002'

# (canonical) fMRI response time series
fpath = f"{proj_dir}/nb/apr22/subj{subj}_uncens.xmat.1D"

rawCols = open(fpath, 'r').readlines()[3].split('"')[1].split(';')
rawCols = [i.strip() for i in rawCols]
# using only regressors of interest
# removing 40 polort regressors and 6 motion regressors
useCols = [i for i in rawCols if "Run" not in i if "Motion" not in i]
# print(len(useCols), useCols)

df = pd.DataFrame(np.loadtxt(fpath), columns=rawCols)
df.columns
X_variables = df[useCols]

num_blocks = 6 * 2 # 2 blocks within each run
l = len(X_variables) // (num_blocks) # window length of a block

appr_regs = [X_variables['closeAppr#0'].iloc[l*idx_block:l*(idx_block+1)].values for idx_block in np.arange(num_blocks)]
retr_regs = [X_variables['closeRetr#0'].iloc[l*idx_block:l*(idx_block+1)].values for idx_block in np.arange(num_blocks)]


# ## stimulus time series

# In[5]:


# stimulus time series
proj_path = "/home/govindas/vscode-BSWIFT-mnt"
proximity_file = f"{proj_path}/eCON/dataset/preproc2/CON{subj}/regs_fancy/CON{subj}_all_regs.txt"
df_prox = pd.read_csv(proximity_file, delimiter='\t')
proximity = df_prox['prox'].values

# censor15_file = f"{proj_path}/eCON/dataset/preproc2/CON{subj}/regs_fancy/CON{subj}_censor15.txt"
# censor15 = np.loadtxt(censor15_file)
censor_prox = proximity != 0.0

tr = 1.25
TR_ref = np.arange(0,2*l*tr,tr)
def return_closest_TR(timings):
    """
    Parameter
    ---------
    timings: numpy.1darray of stim onsets
    
    Returns
    -------
    tuple of numpy.1darray,
        TR_time, TRs = TR times that closesly match with the stim onsets, TR number
    """
    TR_time = []; TRs = []
    for n in timings:
        time = min(TR_ref, key=lambda x:abs(x-n))
        TR_time.append(time)
        TRs.append(np.where(TR_ref==time)[0][0])
    return np.array(TR_time), np.array(TRs)

def get_stimulus_time_series(stim_file):
    with open(stim_file, 'r') as f:
        stim_list = f.read().split('\n')

    stimulus = np.zeros(shape=(len(df),))
    for idx_run, run in enumerate(stim_list):
        events = run.split(' ')
        # starts, durations
        starts_durs = [list(map(float, event.split(':'))) for event in events]
        starts_durs = np.stack(starts_durs, axis=0)
        starts_durs = np.apply_along_axis(return_closest_TR, axis=1, arr=starts_durs)[:, 1, :].astype(int)
        starts, durs = starts_durs[:, 0], starts_durs[:, 1]
        starts += idx_run * 2*l
        for start, dur in zip(starts, durs):
            stimulus[start:start+dur] = 1.0
    stimulus *= censor_prox
    return [stimulus[l*idx_block:l*(idx_block+1)] for idx_block in np.arange(num_blocks)]

appr_file = (f"{proj_path}/eCON/dataset/preproc2/CON{subj}" 
             f"/splitted_regs_fancy/CON{subj}_closeAllApprMR.txt")
retr_file = (f"{proj_path}/eCON/dataset/preproc2/CON{subj}"
             f"/splitted_regs_fancy/CON{subj}_closeAllRetrMR.txt")

appr_stims = get_stimulus_time_series(appr_file)
retr_stims = get_stimulus_time_series(retr_file)

prox_stims = [proximity[l*(idx_block): l*(idx_block+1)] for idx_block in np.arange(num_blocks)]
    


# ## plots

# ### stimulus and fMRI response

# In[6]:


nrows, ncols = num_blocks, 1
figsize = (30*ncols, 5.5*nrows)
fig, axs = plt.subplots(
    nrows=nrows, 
    ncols=ncols, 
    figsize=figsize, 
    sharex=False, 
    sharey=True, 
    dpi=150)
for idx_block in np.arange(num_blocks):
    ax = axs[idx_block]
    ax.set_title(f"Run {(idx_block)//2}, Block {idx_block%2}")

    # appr regressor
    ax.plot(appr_stims[idx_block], color='peru', linestyle='-.', linewidth=3, label='appr_stim')
    
    # appr response    
    # a = np.ma.masked_array(appr_regs[idx_block], mask=~appr_segs)
    # ax.plot(a, color='salmon', linestyle='-', linewidth=3, label='appr_resp')
    ax.plot(appr_regs[idx_block], color='salmon', linestyle='-', linewidth=3, label='appr_resp')

    # retr regressor
    ax.plot(retr_stims[idx_block], color='slateblue', linestyle='-.', linewidth=3, label='retr_stim')
    
    # retr response    
    # r = np.ma.masked_array(retr_regs[idx_block], mask=~retr_segs)
    # ax.plot(r, color='cornflowerblue', linestyle='-', linewidth=3, label='retr_resp')
    ax.plot(retr_regs[idx_block], color='cornflowerblue', linestyle='-', linewidth=3, label='retr_resp')

    # combined response
    ax.plot(appr_regs[idx_block] + retr_regs[idx_block], color='orchid', linestyle='-', linewidth=3, label='comb_resp')

    # proximity
    ax.plot(prox_stims[idx_block], color='black', linestyle='--', linewidth=3, label='proximity')

    ax.legend()
    ax.grid(True)


# **OBSERVATIONS**
# 1. This plot shows that the (canonical) fMRI response peaks at around 8.75TRs. And the information about the stimulus is present from around 5(or 6)TRs. So we should take segments of the fMRI signal shifted by 5(6)TRs from the onset of stimulus. 
# 
# 2. We also observe that the responses for the two classes (appr/ retr) oscillate. We can define a segment as time points between the intersections of the response curves. This is because at intersection influence of both the stimuli is similar, and at other time points one of them dominates. Thus such a segment would have least *contamination* from the other response.
# 
# 3. In addition to choosing such segments, we can introduce a third target label: **Don't care, or indistinguishable, or unidentifiable**. So the time points between intersections will belong to either of the two classes and the points around intersections will belong to the new third class.
# 

# ### appr, retr, and don't-care time segments

# In[7]:


eps = 0.1
appr_segs, retr_segs = [], []
for idx_block in np.arange(num_blocks):
    appr_segs.append((appr_regs[idx_block] > retr_regs[idx_block] + eps) & (appr_regs[idx_block] > 0.5))
    retr_segs.append((retr_regs[idx_block] > appr_regs[idx_block] + eps) & (retr_regs[idx_block] > 0.5))

appr_segs = np.concatenate(appr_segs)
retr_segs = np.concatenate(retr_segs)
dont_care_segs = 1.0 - (appr_segs + retr_segs)

appr_contig_regions = contiguous_regions(appr_segs)
retr_contig_regions = contiguous_regions(retr_segs)
dont_care_contig_regions = contiguous_regions(dont_care_segs)

with open(f"{proj_dir}/nb/apr22/02-contiguous_regions.pkl", 'wb') as f: 
    pickle.dump(
        {'appr_segs': appr_segs,
        'retr_segs': retr_segs,
        'dont_care_segs': dont_care_segs,
        'appr_contig_regions': appr_contig_regions, 
        'retr_contig_regions': retr_contig_regions, 
        'dont_care_contig_regions': dont_care_contig_regions,
        'censor_prox': censor_prox}, 
        f)


# In[8]:


with open(f"{proj_dir}/nb/apr22/02-contiguous_regions.pkl", 'rb') as f:
    stim_dict = pickle.load(f)


# In[9]:


nrows, ncols = 1, 3
figsize = (8*ncols, 3*nrows)
fig, axs = plt.subplots(
    nrows=nrows, 
    ncols=ncols, 
    figsize=figsize, 
    sharex=False, 
    sharey=True, 
    dpi=150)

ax = axs[0]
ax.set_title(f"Appr segments")
ax.hist(segment_lengths(appr_segs, appr_contig_regions))
ax.set_ylabel(f"bin counts")
ax.set_xlabel(f"segment length")

ax = axs[1]
ax.set_title(f"Retr segments")
ax.hist(segment_lengths(retr_segs, retr_contig_regions))
ax.set_xlabel(f"segment length")

ax = axs[2]
ax.set_title(f"Don't-care segments")
ax.hist(segment_lengths(dont_care_segs, dont_care_contig_regions))
ax.set_xlabel(f"segment length")


# We plan to partition fMRI time series into three categories of contiguous segments: (uncontaminated) appr, (uncontaminated) retr, and unidentifiable/ don't-care. 
# 
# **Till now**, we found the indices of these contiguous time periods in `appr_segs`, `retr_segs`, and `dont_care_segs` ndarrays. We used the canonical fMRI responses to do so. 
# 
# **Importantly**, many of the appr/ retr segments are 6 to 7 TRs long. We have sufficient data samples to use a RNN model.

# ### partitions of the canonical fMRI response 

# In[10]:


nrows, ncols = num_blocks, 1
figsize = (30*ncols, 5.5*nrows)
fig, axs = plt.subplots(
    nrows=nrows, 
    ncols=ncols, 
    figsize=figsize, 
    sharex=False, 
    sharey=True, 
    dpi=150)
for idx_block in np.arange(num_blocks):
    ax = axs[idx_block]
    ax.set_title(f"Run {(idx_block)//2}, Block {idx_block%2}")
    
    # appr response    
    a = np.ma.masked_array(appr_regs[idx_block], mask=~appr_segs[l*idx_block:l*(idx_block+1)])
    ax.plot(a, color='salmon', linestyle='-', linewidth=3, label='appr_resp')
    ax.plot(appr_regs[idx_block], color='salmon', linestyle='-', linewidth=3, label='appr_resp', alpha=0.3)
    
    # retr response    
    r = np.ma.masked_array(retr_regs[idx_block], mask=~retr_segs[l*idx_block:l*(idx_block+1)])
    ax.plot(r, color='cornflowerblue', linestyle='-', linewidth=3, label='retr_resp')
    ax.plot(retr_regs[idx_block], color='cornflowerblue', linestyle='-', linewidth=3, label='retr_resp', alpha=0.3)

    # dont-care points
    ax.plot(dont_care_segs[l*idx_block:l*(idx_block+1)], color='black', linestyle='--', linewidth=3, label="don't-care", alpha=0.5)

    # proximity
    ax.plot(prox_stims[idx_block], color='seagreen', linestyle='-', linewidth=3, label='proximity')

    ax.legend()
    ax.grid(True)

