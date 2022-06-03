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
# import shap

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
month_dir = f"{proj_dir}/nb/mar22"

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
# shap.initjs()


def get_data_samples(data, subject_list, mask_value=-1.0):
    X = [] 
    y = []
    Xlen = [] 
    pos = [] 
    idx = []

    for subject in subject_list:
        X += data[subject][0]
        y += data[subject][1]
        Xlen += [i.shape[0] for i in data[subject][0]]
        pos += data[subject][2]
        idx += data[subject][3]

    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X, padding="post",
        dtype='float', value=mask_value
    )

    y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        y, padding="post",
        dtype='float', value=mask_value
    )

    return X_padded, y_padded, Xlen, pos, idx

def get_segments(X, y, y_pred, val, ):
    fMRI_tss, segments, preds = [], [], []
    time_steps = np.where(y == val)[0]
    segs = np.split(time_steps, np.where(np.diff(time_steps) != 1.0)[0]+1)
    for seg in segs:
        fMRI_tss.append(X[seg, :])
        segments.append(y[seg])
        preds.append(y_pred[seg, int(val)])
    return fMRI_tss, segments, preds

def get_masked_array(sequences:list, mask_value=-1.0):
    for idx, sequence in enumerate(sequences):
        sequence = tf.keras.preprocessing.sequence.pad_sequences(
            sequence,
            maxlen=None,
            dtype='float32',
            padding='post',
            value=mask_value)
        sequence = np.ma.masked_array(sequence, mask=(sequence == mask_value))
        sequences[idx] = np.ma.masked_invalid(sequence)
    return sequences
    
def get_appr_retr_segment_data(Xs, ys, y_preds):
    appr_lists = [[] for _ in range(3)] # appr_fMRI_tss, appr_segments, appr_preds
    retr_lists = [[] for _ in range(3)] # retr_fMRI_tss, retr_segments, retr_preds
    appr, retr = 1.0, 0.0
    mask_value = -1.0
    for piece_idx in np.arange(Xs.shape[0]):
        X = Xs[piece_idx, :, :]
        # time steps that are not masked
        time_steps_unmasked = np.unique(np.where(X != mask_value)[0])

        # get unmasked targets
        y = ys[piece_idx, time_steps_unmasked]
        y_pred = y_preds[piece_idx, time_steps_unmasked, :]

        # find continuous segments of all classes
        appr_lists = list(map(add, appr_lists, get_segments(X, y, y_pred, appr)))
        retr_lists = list(map(add, retr_lists, get_segments(X, y, y_pred, retr)))

    # padding and masking the sequences
    appr_lists = get_masked_array(appr_lists)
    retr_lists = get_masked_array(retr_lists)

    # create mean and std of the sequences
    stats = {'appr':[], 'retr':[]}
    for arr in appr_lists:
        stats['appr'].append(arr.mean(axis=0))
        stats['appr'].append(1.96 * arr.std(axis=0) / np.sqrt(np.sum(~arr.mask, axis=0)))
    for arr in retr_lists:
        stats['retr'].append(arr.mean(axis=0))
        stats['retr'].append(1.96 * arr.std(axis=0) / np.sqrt(np.sum(~arr.mask, axis=0)))
    
    return stats

def plot_roi_ts(ts, nw, masker_labels, nrows, ncols, figsize):

    fig, axs = plt.subplots(nrows=nrows, 
                            ncols=ncols, 
                            figsize=figsize, 
                            sharex=True, 
                            sharey=True, 
                            dpi=150)
    
    for idx_roi, roi in enumerate(nw):
        ax = axs[idx_roi//ncols, np.mod(idx_roi,ncols)]

        '''
        approach
        '''
        ts_mean, ts_std, _, _, preds_mean, preds_std = ts['appr']
        # time series
        time = np.arange(ts_mean.shape[0])
        ax.plot(
            time, ts_mean[:, roi], 
            color='orange', label='appr')
        ax.fill_between(
            time, 
            (ts_mean[:, roi] - ts_std[:, roi]), 
            (ts_mean[:, roi] + ts_std[:, roi]), 
            alpha=0.3, color='orange')
        # prediction
        ax.plot(
            time, preds_mean, 
            color='red', label='appr_pred')
        ax.fill_between(
            time, 
            (preds_mean - preds_std), 
            (preds_mean + preds_std), 
            alpha=0.3, color='red')

        '''
        retreat
        '''
        # time series 
        ts_mean, ts_std, _, _, preds_mean, preds_std = ts['retr']
        time = np.arange(ts_mean.shape[0])
        ax.plot(time, ts_mean[:, roi], color='green', label='retr')
        ax.fill_between(time, 
                        (ts_mean[:, roi] - ts_std[:, roi]), 
                        (ts_mean[:, roi] + ts_std[:, roi]), 
                        alpha=0.3, color='green')
        # prediction
        ax.plot(
            time, preds_mean, 
            color='blue', label='retr_pred')
        ax.fill_between(
            time, 
            (preds_mean - preds_std), 
            (preds_mean + preds_std), 
            alpha=0.3, color='blue')

        ax.axhline(0.5, time[0], time[-1], color='black', linestyle='--')

        '''
        layout
        '''
        ax.set_title(f"{masker_labels[roi]}")
        if idx_roi//ncols == nrows-1: ax.set_xlabel('Time(TR)')
        if np.mod(idx_roi,ncols) == 0: ax.set_ylabel('Activity')
        ax.set_xticks(time)
        ax.set_xticklabels(time+1)
        ax.set_ylim(-1.0, 1.0)
        ax.grid(True)
        ax.legend()