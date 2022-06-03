#!/usr/bin/env python
# coding: utf-8

# # May 11-14,19, 2022: Idea1: don't care labels: create and train a model
# 
# Write your own `loss_function` to ignore padded time steps while computing loss and training/evaluating. 

# In[1]:


import os
import sys
from os.path import join as pjoin

import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
import tensorflow.keras.backend as K
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
month_dir = f"{proj_dir}/nb/may22"

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


# ## training and testing datasets

# In[2]:


def get_data(data, subject_list):
    X = [] 
    y = []
    Xlen = [] 
    global MASK

    for subject in subject_list:
        X += data[subject]['X']
        y += data[subject]['y']
        Xlen += [i.shape[0] for i in data[subject]['X']]

    X_padded = tf.keras.preprocessing.sequence.pad_sequences(
        X, 
        padding="post",
        dtype='float', 
        value=MASK
    )
    
    y_padded = tf.keras.preprocessing.sequence.pad_sequences(
        y, 
        padding="post",
        dtype='float',
        value=MASK
    )
    # y_padded = tf.convert_to_tensor(y)

    return X_padded, y_padded, Xlen


# In[3]:


'''
target stimuli
'''
APPR, RETR = 1, 0
MASK = 0.5

data_file = f"{proj_dir}/data/emoprox2/idea1_data.pkl"
with open(data_file, 'rb') as f:
    data = pickle.load(f)

'''
(hyper)-parameters
'''
class ARGS(): pass
args = ARGS()

args.SEED = 74

# data args
args.num_subjects = len(data.keys())
args.num_train = args.num_subjects // 2
args.num_test = args.num_subjects - args.num_train

# model args
args.num_units = 32 #16 #32
args.num_classes = 2 # for binary classification
args.l2 = 1e-2
args.dropout = 0.0 #0.8
args.learning_rate = 4e-4

args.num_epochs = 50
args.validation_split = 0.2
args.batch_size = 64

args.return_sequences = True

'''
generate dataset for the model
'''
subject_list = list(data.keys())
random.Random(args.SEED).shuffle(subject_list)

train_list = subject_list[:args.num_train]
test_list = subject_list[args.num_train:]

(X_train, y_train, 
 len_train) = get_data(data, train_list)
(X_test, y_test, 
 len_test) = get_data(data, test_list)

print(X_train.shape, y_train.shape)


# ## model: GRU

# In[4]:


class CustomSCCE(tf.keras.losses.Loss):
    '''
    Custom Sparse Categorical Crossentropy
    '''
    def __init__(self, name='custom_scce'):
        super().__init__(name=name)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, 
            reduction=tf.keras.losses.Reduction.NONE
        )
    
    def call(self, y_true, y_pred):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(y_true != MASK, tf.float32)
        loss *= mask
        # average on non-zeros
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)


# In[5]:


tf.random.set_seed(args.SEED)
regularizer = tf.keras.regularizers.l2(l2=args.l2) 
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

model_file = f"{results_dir}/emoprox_full_data/models/GRU_classifier_gruunits_{args.num_units}_idea1"
history_file = f"{results_dir}/emoprox_full_data/models/GRU_classifier_gruunits_{args.num_units}_idea1_history"

'''
build model
'''
model = model_definitions.get_GRU_classifier_model(
    X_train, 
    args, 
    regularizer, 
    mask_value=MASK, 
    return_sequences=True)

'''
compile model
'''
model.compile(
    loss=CustomSCCE(),
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

if os.path.exists(f"{model_file}.index"):
    '''
    load weights
    '''
    model.load_weights(model_file)
    history = json.load(open(f"{history_file}", 'r'))
else:
    '''
    train model
    '''
    then = time.time()
    history = model.fit(
        x=X_train, 
        y=y_train,
        batch_size=args.batch_size, 
        epochs=args.num_epochs, 
        verbose=1,
        callbacks=tf.keras.callbacks.EarlyStopping(patience=5),
        validation_split=args.validation_split, 
        shuffle=True)
    print('--- train time =  %0.4f seconds ---' %(time.time() - then))

    '''
    save model weights
    '''
    model.save_weights(model_file)
    history = history.history
    json.dump(history, open(f"{history_file}", 'w'))

# evaluate the model
eval_hist = model.evaluate(X_test, y_test)


# In[20]:


model.summary()


# In[6]:


fig, axs = plt.subplots(
    nrows=2, ncols=1,
    figsize=(11,5),
    dpi=150
)

ax = axs[0]
ax.plot(history['loss'], color='tomato', linestyle='-.', label='training_loss')
ax.plot(history['val_loss'], color='forestgreen', label='valid_loss')
ax.set_ylabel(f"losses")
ax.set_xlabel(f"epochs")
ax.legend()
ax.grid(True)

ax = axs[1]
ax.plot(history['sparse_categorical_accuracy'], color='tomato', linestyle='-.', label='training_acc')
ax.plot(history['val_sparse_categorical_accuracy'], color='forestgreen', label='valid_acc')
ax.set_ylabel(f"accuracies")
ax.set_xlabel(f"epochs")
ax.legend()
ax.grid(True)


# `training loss` is more than `validation loss`, and `training accuracy` is less than `validation loss`!!!
# this might be because:
# 1. underfitting: model is not able to learn training data distribution
# 2. ???
# 
# And both accuracies are not more than 60%. This may imply model is **underfitting**.
# 1. Check with training null datasets.
# 2. Check if the evaluation metric if correctly computing. Check at which timesteps the model fails to predict correctly. Give some samples or pass the entire timeseries for a run/ block of a subject.
# 3. Compare with other (simpler) models like logistic regression, svm, tcn etc. 

# ### 1. null datasets and null distribution of accuracies.

# In[ ]:


num_null_models = 10
null_eval_hists_file = (
    f"{results_dir}/emoprox_full_data/models"
    f"/GRU_classifier_null_eval_hists_gruunits_{args.num_units}_idea1")
if not os.path.exists(null_eval_hists_file):
    null_eval_hists = []
    for idx_null in tqdm(np.arange(num_null_models)):
        # build, train, and save the model
        '''
        build model
        '''
        null_model = model_definitions.get_GRU_classifier_model(
            X_train, 
            args, 
            regularizer, 
            mask_value=MASK, 
            return_sequences=True
        )

        '''
        train model
        '''
        null_model.compile(
            loss=CustomSCCE(),
            optimizer=optimizer,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

        then = time.time()
        null_history = null_model.fit(
            x=X_train, 
            y=np.random.randint(low=0, high=2, size=(y_train.shape)).astype(float),
            batch_size=args.batch_size, 
            epochs=args.num_epochs, 
            verbose=1,
            callbacks=tf.keras.callbacks.EarlyStopping(patience=5),
            validation_split=args.validation_split, 
            shuffle=True)
        print('--- train time =  %0.4f seconds ---' %(time.time() - then))

        # evaluate the model
        null_eval_hists.append(null_model.evaluate(X_test, y_test, verbose=0)[1])
    
    # save null evaluations
    with open(null_eval_hists_file, 'wb') as f:
        pickle.dump(null_eval_hists, f)
else:
    # load null evaluations
    with open(null_eval_hists_file, 'rb') as f:
        null_eval_hists = pickle.load(f) 


# In[ ]:


plt.hist(np.stack(null_eval_hists, axis=0))
plt.axvline(eval_hist[1], color='red')
plt.xlabel('accuracies')
plt.ylabel('bin counts')
plt.title('accuracy histograms of null models (blue) and actual model (red)')


# ### 2. check evaluation metric.

# evaluation metric seems okay. just that it also considers the masked timesteps. This is not a major concern currently. 
# 
# The concern is trained model predicts only the first label, i.e. `y_pred[:,0]`. `y_pred[:, 0] > 0.0 > y_pred[:, 1]`. **This is a serous concern!!** 

# In[7]:


num_appr = np.sum(y_train[:, 0] == 1.0)
num_retr = np.sum(y_train[:, 0] == 0.0)
print(num_appr, num_retr)

y_pred = model(X_train)
print(y_pred[20:40, ...])
np.sum(np.equal(y_train, np.argmax(y_pred, axis=-1))) / (y_train.shape[0] * y_train.shape[1])


# In[8]:


random_model = model_definitions.get_GRU_classifier_model(
    X_train,
    args,
    regularizer,
    mask_value=MASK,
    return_sequences=True
)
random_model.compile(
    loss=CustomSCCE(),
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)
y_pred = random_model(X_train)
print(np.argmax(y_pred, axis=-1)[20:40, ...])
np.sum(np.equal(y_train, np.argmax(y_pred, axis=-1))) / (y_train.shape[0] * y_train.shape[1])


# ### 3. training other models

# #### logistic regression

# In[12]:


X = X_train
mask_value = MASK

tf.random.set_seed(args.SEED)
'''
build model
'''
inputs = tf.keras.Input(
    shape=(None, X.shape[-1]),
    name='input'
)
mask_layer = tf.keras.layers.Masking(
    mask_value=mask_value, 
    input_shape=[None, inputs.shape[-1]], 
    name='masking')
x = mask_layer(inputs)

intermediate_layer = tf.keras.layers.Dense(
    args.num_units,
    activation='relu'
)
x = intermediate_layer(x)

output_layer = tf.keras.layers.Dense(
        args.num_classes, 
        activation=None, 
        name='output')
outputs = output_layer(x)
log_reg = tf.keras.Model(inputs=inputs, outputs=outputs, name='logistic_regression')
'''
compile model
'''
log_reg.compile(
    loss=CustomSCCE(),
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

print(log_reg.summary())

'''
train model
'''
log_reg.fit(
    x=X_train, 
    y=y_train,
    batch_size=args.batch_size, 
    epochs=args.num_epochs, 
    verbose=1,
    callbacks=tf.keras.callbacks.EarlyStopping(patience=5),
    validation_split=args.validation_split, 
    shuffle=True
)
'''
evaluate model
'''
eval_hist = log_reg.evaluate(X_test, y_test)


# In[13]:


y_pred = log_reg(X_train)
print(y_pred[20:40, ...])


# #### temporal convolutional network: tcn

# In[26]:


args.num_filters = 32 # MAY NEED TO CHANGE
args.window_size = 4
args.strides = 1


# In[27]:


X = X_train
mask_value = MASK

tf.random.set_seed(args.SEED)
'''
build model
'''
inputs = tf.keras.Input(
    shape=(None, X.shape[-1]),
    name='input'
)
mask_layer = tf.keras.layers.Masking(
    mask_value=mask_value, 
    input_shape=[None, inputs.shape[-1]], 
    name='masking')
x = mask_layer(inputs)

intermediate_layer = tf.keras.layers.Conv1D(
    filters=args.num_filters,
    kernel_size=args.window_size,
    strides=args.strides,
    padding='same', # MAY CHANGE TO `CAUSAL`/ `SAME`/ `VALID`
    activation='relu',
    name='conv'
)
x = intermediate_layer(x)

output_layer = tf.keras.layers.Dense(
        args.num_classes, 
        activation=None, 
        name='output')
outputs = output_layer(x)
tcn = tf.keras.Model(inputs=inputs, outputs=outputs, name='tcn')
'''
compile model
'''
tcn.compile(
    loss=CustomSCCE(),
    optimizer=optimizer,
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

print(tcn.summary())

'''
train model
'''
tcn.fit(
    x=X_train, 
    y=y_train,
    batch_size=args.batch_size, 
    epochs=args.num_epochs, 
    verbose=1,
    callbacks=tf.keras.callbacks.EarlyStopping(patience=5),
    validation_split=args.validation_split, 
    shuffle=True
)
'''
evaluate model
'''
eval_hist = tcn.evaluate(X_test, y_test)


# In[28]:


y_pred = tcn(X_train)
print(y_pred[20:40, ...])


# #### Discussion:
# Somehow all these models are stuck at 55% accuracy. **This points to how loss and accuracy are calculated.**
