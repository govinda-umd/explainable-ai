import numpy as np
from tqdm import tqdm

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def get_max_data_trials(args, data_df, subj_idx_list):
    normalize = lambda z: (z - np.mean(z)) / np.std(z)

    X, y = [], []
    for idx_row in tqdm(subj_idx_list):
        subj, ts, targets = data_df.iloc[idx_row]

        for label in args.LABELS:
            contig_regions = contiguous_regions(targets == label)

            for region in contig_regions:
                X.append(ts[region[0]: region[1], args.roi_idxs])
                y.append(targets[region[0]: region[1]])

    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)

    for idx_roi in np.arange(X.shape[-1]):
        X[:, :, idx_roi] = normalize(np.squeeze(X[:, :, idx_roi]))
    
    return X, y #torch.FloatTensor(X).to(device=device), torch.LongTensor(y).to(device=device)


def sim_data_additive_white_noise(args, X, y):
    '''
    white-noise 
    -----------
    1. std's for each roi and each tp
    2. simulated data with i.i.d. normal noise (white noise) around mean time series
    '''
    X_, y_ = [], []
    for label in args.LABELS:
        idx = y[:, 0] == label
        X_ += [np.random.normal(
            loc=np.mean(X[idx], axis=0), 
            scale=args.noise_level*np.std(X[idx], axis=0), 
            size=X[idx].shape
        )]
        y_ += [np.ones(shape=(X[idx].shape[:-1])) * label]

    X_ = np.concatenate(X_, axis=0)
    y_ = np.concatenate(y_, axis=0)

    perm = np.random.permutation(y_.shape[0])
    X_ = X_[perm]
    y_ = y_[perm]

    return X_, y_