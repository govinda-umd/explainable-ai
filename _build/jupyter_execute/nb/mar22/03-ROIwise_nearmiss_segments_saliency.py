#!/usr/bin/env python
# coding: utf-8

# # March 11, 14-15, 2022: saliency maps

# In[1]:


import os
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
import tensorflow as tf
# import tensorflow_addons as tfa
import pickle, time, random
import neural_structured_learning as nsl
from tqdm import tqdm
import json

# explanation tools
import shap

# plotting
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
shap.initjs()


# ## data 

# In[2]:


def get_data_samples_actual(data, subject_list):
    
    X = []; y = []
    target = np.expand_dims(np.array([0,0,0,0,0,0,0,1,1,1,1,1]),axis=0).astype(np.float64)
    
    for subject in subject_list:
        
        num_samples = data[subject].shape[0]
        X.append(data[subject])
        y.append(np.repeat(target, num_samples, axis=0))

    return np.vstack(X), np.vstack(y)

def get_data_samples_avg(data, subject_list):
    
    X = []; y = []
    target = np.expand_dims(np.array([0,0,0,0,0,0,0,1,1,1,1,1]),axis=0).astype(np.float64)
    
    for subject in subject_list:
        
        num_samples = data[subject].shape[0]
        X.append(np.expand_dims(np.mean(data[subject], axis=0), axis=0))
        y.append(target)

    return np.vstack(X), np.vstack(y)


# In[3]:


raw_data_file = f"{proj_dir}/data/classification_data/MAX_rois_122subjs_nearmiss_segments_withoutshock.pkl"
with open(raw_data_file, 'rb') as f:
    data = pickle.load(f)


# ### parameters

# In[4]:


class ARGS(): pass
args = ARGS()

args.SEED = 74

# data
args.num_subjects = len(data.keys())
args.num_train = args.num_subjects // 2
args.num_test = args.num_subjects - args.num_train


# ### organizing into tensors

# In[5]:


subject_list = list(data.keys())
random.Random(args.SEED).shuffle(subject_list)

train_list = subject_list[:args.num_train]
test_list = subject_list[-args.num_test:]

X_train, y_train = get_data_samples_actual(data, train_list)
X_test, y_test = get_data_samples_actual(data, test_list)

X_train = tf.convert_to_tensor(X_train)
y_train = tf.convert_to_tensor(y_train)
X_test = tf.convert_to_tensor(X_test)
y_test = tf.convert_to_tensor(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# ## model

# In[6]:


# model
args.num_units = 32
args.num_classes = 2 # for binary classification
args.l2 = 1e-2
args.dropout = 0.8
args.learning_rate = 4e-4

args.num_epochs = 100
args.validation_split = 0.2
args.batch_size = 64

# multiplier to adversarial regularization loss. Defaults to 0.2. 
args.adv_multiplier = 1
# step size to find the adversarial sample. Defaults to 0.001. 
args.adv_step_size = 3
# type of tensor norm to normalize the gradient. Defaults to L2 norm. 
# Input will be converted to NormType when applicable 
# (e.g., a value of 'l2' will be converted to nsl.configs.NormType.L2). 
args.adv_grad_norm = 'l2'

args.temp = 20


# In[7]:


regularizer = tf.keras.regularizers.l2(l2=args.l2) 
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)


# ### loading (non-adversarial) model

# In[8]:


tf.random.set_seed(args.SEED)

model_file = f"{results_dir}/models/GRU_classifier_model"
history_file = f"{results_dir}/models/GRU_classifier_model_history"
if os.path.exists(model_file):
    # load the model
    model = tf.keras.models.load_model(model_file)
    history = json.load(open(f"{history_file}", 'r'))
else:
    print(f"the model {model_file} was not saved")

model.evaluate(X_test, y_test)

tf.keras.utils.plot_model(
    model, 
    f"figures/GRU_classifier_model.png", 
    show_shapes=True)

# model.summary()


# ### adversarial-regularized model

# In[9]:


adv_model_file = f"{results_dir}/models/GRU_classifier_adv_model"
adv_history_file = f"{results_dir}/models/GRU_classifier_model_adv_history"
if os.path.exists(model_file):
    # load the model
    adv_model = tf.keras.models.load_model(adv_model_file)
    adv_history = json.load(open(adv_history_file, 'r'))
else:
    print(f"the model {adv_model_file} was not saved")

adv_model.evaluate(x={'input': X_test, 'label': y_test})

# tf.keras.utils.plot_model(
#     adv_model, 
#     f"figures/GRU_classifier_adv_model.png", 
#     show_shapes=True)

adv_model.summary()


# ## methods for interpreting model predictions

# ### add temperature to pretrained model

# In[10]:


T_layer = tf.keras.layers.Lambda(lambda x: x / args.temp, name='temperature')
S_layer = tf.keras.layers.Softmax(axis=-1, name='softmax')


# In[11]:


model.trainable = False

t_model = tf.keras.Sequential(
    model.layers +\
    [T_layer, S_layer])

t_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

t_model.evaluate(X_test, y_test)

t_model.summary()

tf.keras.utils.plot_model(
    t_model, 
    f"figures/GRU_classifier_temp_model.png", 
    show_shapes=True)


# In[12]:


adv_model.trainable = False

t_adv_model = tf.keras.Sequential(
    adv_model.layers +\
    [T_layer, S_layer])

t_adv_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

t_adv_model.evaluate(X_test, y_test)
y_pred = t_adv_model.predict(X_test)
# plt.hist(y_pred.flatten(), bins=1000)

t_adv_model.summary()

tf.keras.utils.plot_model(
    t_adv_model, 
    f"figures/GRU_classifier_temp_adv_model.png", 
    show_shapes=True)


# ### saliency
# We compute saliency of input features (ROIs) based on the temperature scaled model. This is because the temperature scaling brings the last layer's activity closer to zero so that gradient through the softmax layer becomes non-zero. 

# In[13]:


def compute_saliency_maps(model, fMRI_sequence, target_class_idx):
    '''
    compute saliency maps:
    1> pass fMRI_sequence to get class scores
    2> backpropagate class score of target class
    3> return gradients w.r.t. inputs
    '''
    target = np.expand_dims(np.array([0,0,0,0,0,0,0,1,1,1,1,1]),axis=0).astype(np.float64)
    fMRI_sequence = tf.convert_to_tensor(fMRI_sequence)
    with tf.GradientTape() as tape:
        tape.watch(fMRI_sequence)
        probs = model(fMRI_sequence)[:, :, target_class_idx] # class probability score of the target class
    return tape.gradient(probs, fMRI_sequence).numpy()


# In[14]:


def unmask_ROIwise(x, mask):
    '''
    input:
    -----
    x: np.array of size <num_time, num_rois>
    mask: nilearn niimg of size <num_vox_x, num_vox_y, num_vox_z>
    
    output:
    ------
    final_mask: nilearn niimg that contains the unmasked 4D 
                volumetric data of x
    '''
    mask_data = mask.get_fdata().copy()
    s = list(mask_data.shape) + list(x.shape[:-1])
    final_data = np.zeros(shape=s, dtype=np.float32)
    
    mask_data = np.expand_dims(mask_data, axis=-1)
    num_rois = np.int(np.max(mask_data))
    
    for idx_roi in np.arange(num_rois):
        
        roi_data = x[:, idx_roi] * (mask_data==(idx_roi+1))
        final_data += roi_data
        
    final_mask = image.new_img_like(ref_niimg=mask, 
                              data=final_data, 
                              copy_header=False)
    return final_mask
        

! rm nifti_files/subject_saliency/*
# In[17]:


normalize = lambda a: (a - np.mean(a)) / np.std(a)
appr_TR = 7
saliency_appr = []
saliency_retr = []
for idx_sample in tqdm(range(X_test.shape[0])):

    sample = np.expand_dims(X_test[idx_sample, :, :], axis=0)
    
    # approach class
    sample_saliency = compute_saliency_maps(
        model=t_model, 
        fMRI_sequence=sample, 
        target_class_idx=0)
    sample_saliency = normalize(sample_saliency)
    saliency_appr.append(sample_saliency)
    
    # retreat class
    sample_saliency = compute_saliency_maps(
        model=t_model, 
        fMRI_sequence=sample, 
        target_class_idx=1)
    sample_saliency = normalize(sample_saliency)
    saliency_retr.append(sample_saliency)

mean_saliency_appr = np.mean(np.vstack(saliency_appr), axis=0)
mean_saliency_retr = np.mean(np.vstack(saliency_retr), axis=0)
# saliency_contrast = mean_saliency_appr - mean_saliency_retr

# sal_nii_list = []
# for t in range(mean_saliency_appr.shape[0]):
#     sal_nii_list.append(unmask_ROIwise(saliency_contrast[t], MAX85))
# mean_saliency_appr = concat_imgs(sal_nii_list)
# mean_saliency_appr.to_filename('nifti_files/MAX85_sal_contrast.nii.gz') 


# In[18]:


MAX85 = image.load_img(img=f"/home/govindas/parcellations/MAX_85_ROI_masks/MAX_ROIs_final_gm_85.nii.gz")
sal_appr_niimg = unmask_ROIwise(mean_saliency_appr, MAX85)
sal_retr_niimg = unmask_ROIwise(mean_saliency_retr, MAX85)
sal_appr_niimg.to_filename(f"{month_dir}/figures/saliency_approach.nii.gz")
sal_retr_niimg.to_filename(f"{month_dir}/figures/saliency_retreat.nii.gz")

np.mean(np.vstack(subj_saliency_values), axis=0)%%bash 

target="nifti_files/subject_saliency"
apprfile="$target/CON???_appr_sal.nii.gz"
retrfile="$target/CON???_retr_sal.nii.gz"

MAX85="/home/joyneelm/approach-retreat/data/processed/masks/MAX_ROIs_final_gm_85.nii.gz"

3dttest++ -overwrite \
-setA "$apprfile" \
-setB "$retrfile" \
-labelA appr_saliency \
-labelB retr_saliency \
-prefix "nifti_files/MAX85_appr_retr_sal_contrast.nii.gz" -mask "$MAX85" -pairednormalize = lambda a: (a - np.mean(a)) / np.std(a)
MAX85 = load_img(img='/home/joyneelm/approach-retreat/data/processed/masks/MAX_ROIs_final_gm_85.nii.gz')
saliency_appr = []
saliency_retr = []
for idx_sample in tqdm(range(X_test.shape[0])):

    sample = np.expand_dims(X_test[idx_sample, :, :], axis=0)
    
    # approach class
    sample_saliency = compute_saliency_maps(t_model, sample, 0)
    sample_saliency = normalize(sample_saliency)
    saliency_appr.append(sample_saliency)
    
    # retreat class
    sample_saliency = compute_saliency_maps(t_model, sample, 1)
    sample_saliency = normalize(sample_saliency)
    saliency_retr.append(sample_saliency)

mean_saliency_appr = np.mean(np.vstack(saliency_appr), axis=0)
mean_saliency_retr = np.mean(np.vstack(saliency_retr), axis=0)
saliency_contrast = mean_saliency_appr - mean_saliency_retr

sal_nii_list = []
for t in range(mean_saliency_appr.shape[0]):
    sal_nii_list.append(unmask_ROIwise(saliency_contrast[t], MAX85))
mean_saliency_appr = concat_imgs(sal_nii_list)
mean_saliency_appr.to_filename('nifti_files/MAX85_sal_contrast.nii.gz')        # normalize = lambda a: (a - np.mean(a)) / np.std(a)
# saliency = [[],[]]
# for idx_subj in tqdm(range(X_test.shape[0])):
#     for class_label in range(2):
        
#         sample = X_test[idx_subj, :, :]
#         sample = tf.expand_dims(sample, axis=0)

#         sample_saliency = compute_saliency_maps(model, sample, class_label)
#         sample_saliency = normalize(sample_saliency)
        
#         if class_label == 0:        
#             saliency[class_label].append(sample_saliency[:,:7].squeeze().T)
#         else:
#             saliency[class_label].append(sample_saliency[:,7:].squeeze().T)