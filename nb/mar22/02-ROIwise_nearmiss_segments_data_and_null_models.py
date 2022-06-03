#!/usr/bin/env python
# coding: utf-8

# # March 16, 2022: null models

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


# ### defining model

# In[7]:


tf.random.set_seed(args.SEED)

regularizer = tf.keras.regularizers.l2(l2=args.l2) 
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

model_file = f"{results_dir}/models/GRU_classifier_model"
history_file = f"{results_dir}/models/GRU_classifier_model_history"
if os.path.exists(model_file):
    # load the model
    model = tf.keras.models.load_model(model_file)
    history = json.load(open(f"{history_file}", 'r'))
else:
    # build, train, and save the model
    '''
    build model
    '''
    model = get_GRU_classifier_model(
        X_train, 
        args, 
        regularizer)

    '''
    train model
    '''
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    then = time.time()
    history = model.fit(
        x=X_train, 
        y=y_train,
        batch_size=args.batch_size, 
        epochs=args.num_epochs, 
        verbose=1,
        # callbacks=tf.keras.callbacks.EarlyStopping(patience=5),
        validation_split=args.validation_split, 
        shuffle=True)
    print('--- train time =  %0.4f seconds ---' %(time.time() - then))

    '''
    save model
    '''
    model.save(model_file)
    history = history.history
    json.dump(history, open(f"{history_file}", 'w'))
    
# evaluate the model
eval_hist = model.evaluate(X_test, y_test)


# ### null models

# In[ ]:


num_null_models = 100
null_eval_hists = []
for idx_null in tqdm(np.arange(num_null_models)):
    null_model_file = f"{results_dir}/models/GRU_classifier_null_model{idx_null}"
    null_history_file = f"{results_dir}/models/GRU_classifier_null_model{idx_null}_history"
    if os.path.exists(null_model_file):
        # load the model
        null_model = tf.keras.models.load_model(null_model_file)
        null_history = json.load(open(f"{null_history_file}", 'r'))
    else:
        # build, train, and save the model
        '''
        build model
        '''
        null_model = get_GRU_classifier_model(
            X_train, 
            args, 
            regularizer)

        '''
        train model
        '''
        null_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        then = time.time()
        null_history = null_model.fit(
            x=X_train, 
            y=np.random.randint(low=0, high=2, size=(y_train.shape)),
            batch_size=args.batch_size, 
            epochs=args.num_epochs, 
            verbose=0,
            # callbacks=tf.keras.callbacks.EarlyStopping(patience=5),
            validation_split=args.validation_split, 
            shuffle=True)
        print('--- train time =  %0.4f seconds ---' %(time.time() - then))

        '''
        save model
        '''
        null_model.save(null_model_file)
        null_history = null_history.history
        json.dump(null_history, open(f"{null_history_file}", 'w'))

    # evaluate the model
    null_eval_hists.append(null_model.evaluate(X_test, y_test))


# In[ ]:


plt.hist(np.stack(null_eval_hists, axis=0)[:, 1])


# ### adversarial-regularized model

# In[ ]:


model_file = f"{results_dir}/models/GRU_classifier_adv_model"
history_file = f"{results_dir}/models/GRU_classifier_model_adv_history"
if os.path.exists(model_file):
    # load the model
    adv_model = tf.keras.models.load_model(model_file)
    adv_history = json.load(open(history_file, 'r'))
    
else:
    # build, train and save the adv_model
    '''
    base model
    '''
    base_model = get_GRU_classifier_model(
        X_train, 
        args, 
        regularizer)

    '''
    configurations
    '''
    adv_config = nsl.configs.make_adv_reg_config(
        multiplier=args.adv_multiplier,
        adv_step_size=args.adv_step_size,
        adv_grad_norm=args.adv_grad_norm,
    #     pgd_iterations=3,
    #     pgd_epsilon=0.001
    )

    '''
    building adv-model
    '''
    adv_model = nsl.keras.AdversarialRegularization(
        base_model,
        label_keys=['label'],
        adv_config=adv_config
    )

    '''
    training
    '''
    adv_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=optimizer,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    then = time.time()
    adv_history = adv_model.fit(
        x={'input': X_train, 'label': y_train}, 
        epochs=args.num_epochs,
        validation_split=args.validation_split,
        batch_size=args.batch_size,
        verbose=1)
    print('--- train time =  %0.4f seconds ---' %(time.time() - then))

    '''
    saving model
    '''
    tf.keras.models.save_model(adv_model, model_file)
    adv_history = adv_history.history
    json.dump(adv_history, open(f"{history_file}", 'w'))
    
# evaluate the adv. model
adv_eval_hist = adv_model.evaluate(x={'input': X_test, 'label': y_test})


# ### null adversarial models

# In[ ]:


num_null = 100
null_adv_eval_hists = []

for idx_null in tqdm(np.arange(num_null)):
    null_model_file = f"{results_dir}/models/GRU_classifier_null_adv_model{idx_null}"
    null_history_file = f"{results_dir}/models/GRU_classifier_null_model{idx_null}_adv_history"
    if os.path.exists(null_model_file):
        # load the model
        null_adv_model = tf.keras.models.load_model(null_model_file)
        null_adv_history = json.load(open(null_history_file, 'r'))

    else:
        # build, train and save the adv_model
        '''
        base model
        '''
        base_model = get_GRU_classifier_model(
            X_train, 
            args, 
            regularizer)

        '''
        configurations
        '''
        adv_config = nsl.configs.make_adv_reg_config(
            multiplier=args.adv_multiplier,
            adv_step_size=args.adv_step_size,
            adv_grad_norm=args.adv_grad_norm,
        #     pgd_iterations=3,
        #     pgd_epsilon=0.001
        )

        '''
        building adv-model
        '''
        null_adv_model = nsl.keras.AdversarialRegularization(
            base_model,
            label_keys=['label'],
            adv_config=adv_config
        )

        '''
        training
        '''
        null_adv_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

        then = time.time()
        null_adv_history = null_adv_model.fit(
            x={'input': X_train, 'label': np.random.randint(low=0, high=2, 
                                                            size=(y_train.shape))}, 
            epochs=args.num_epochs,
            validation_split=args.validation_split,
            batch_size=args.batch_size,
            verbose=1)
        print('--- train time =  %0.4f seconds ---' %(time.time() - then))

        '''
        saving model
        '''
        tf.keras.models.save_model(null_adv_model, null_model_file)
        null_adv_history = null_adv_history.history
        json.dump(null_adv_history, open(f"{null_history_file}", 'w'))

    # evaluate the adv. model
    null_adv_eval_hists.append(null_adv_model.evaluate(x={'input': X_test, 'label': y_test}))

