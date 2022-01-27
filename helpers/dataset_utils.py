import numpy as np 
import pandas as pd 
from os.path import join, exists, isfile #, isdir, dirname, exists
from glob import glob
import tensorflow as tf


def get_data_df(subj_folder_list, 
                time_series_path, 
                target_path, 
                num_TRs):
    '''
    this organizes the .1D mean time series files and corresponding proximity files into a dataframe.
    the dataframe store the time series split per each run.
    '''
    data_df = pd.DataFrame()
    for subj_folder in subj_folder_list:
        if 'yoked' in subj_folder:
            continue

        subj = subj_folder[-3:]

        input_ts = np.loadtxt(time_series_path.format(subj=subj))
        total_TRs, num_rois = input_ts.shape
        num_runs = total_TRs//num_TRs

        input_ts_run_list = np.split(input_ts, num_runs)
        run_list = list(np.arange(1, num_runs+1))
        subj_list = [subj] * num_runs

        target_df = pd.read_csv(target_path.format(subj=subj), delimiter='\t')
        proximity = target_df['prox'].values
        proximity_run_list = np.split(proximity, num_runs)

        input_censor = target_df['censor'].values
        input_censor_run_list = np.split(input_censor, num_runs)

        tmp_dict = {'subj': subj_list, 
                    'run': run_list, 
                    'ts': input_ts_run_list, 
                    'prox': proximity_run_list, 
                    'censor': input_censor_run_list}
        data_df = pd.concat([data_df, pd.DataFrame(tmp_dict)], ignore_index=True)

    return data_df

def get_Xy(data_df, subjs_list):
    '''
    get the training/testing X and y from the dataframe
    '''
    df = data_df[data_df['subj'].isin(subjs_list)]
    X = df['ts'].tolist()
    y = df['prox'].tolist()
    mask = df['censor'].tolist()

    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)
    mask = np.stack(mask, axis=0)

    return (X, y, mask)

def to_tensor(data_tuple):
    tf_list = []
    for idx, arr in enumerate(data_tuple):
        tf_list.append(tf.convert_to_tensor(arr, dtype=tf.float32))
    return tf_list