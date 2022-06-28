import numpy as np

import torch

def get_attribs(x, target_label, interp_method, model):
    if len(x.shape) == 3:
        attribs = []
        for i in np.arange(x.shape[0]):
            model.zero_grad()
            a = interp_method.attribute(
                inputs=x[i, :, :],
                target=target_label,
                # abs=False
            ).cpu().numpy()
            attribs.append(a)
        attribs = np.stack(attribs, axis=0)
    else:
        attribs = interp_method.attribute(
            inputs=x,
            target=target_label,
            # abs=False
        ).cpu().numpy()
    return attribs


def get_attributions(args, X, y, model, interp_method):
    inputs = {}
    attributions = {} # safe:0, threat:1 
    for label in args.LABELS:
        idx = y[:, 0] == label
        inputs[label] = X[idx, :, :]

        b, t, r = inputs[label].shape
        attributions[label] = np.zeros((len(args.LABELS), b, t, r))
        for target_label in args.LABELS:
            attribs = get_attribs(
                inputs[label],
                target_label,
                interp_method,
                model
            )
            attributions[label][target_label] = attribs
    return attributions



def get_hypotheses(args, label, attributions, compute_null=True):
    N = attributions[label][label].shape[0]
    
    # alternative hypothesis
    H1_mean = np.mean(attributions[label][label], axis=0)
    H1_std = np.std(attributions[label][label], axis=0)

    # null hypothesis
    if compute_null:
        nulls = []
        for _ in np.arange(args.num_null):
            sel = np.random.randint(0, 2, size=(N,))
            null = np.concatenate(
                [
                    attributions[label][label, np.where(sel == label)[0], :, :],
                    attributions[label][1-label, np.where(sel == 1-label)[0], :, :]
                ],
                axis=0
            )
            nulls.append(np.mean(null, axis=0))
        nulls = np.stack(nulls, axis=0)

        H0_mean = np.mean(nulls, axis=0)
        H0_std = np.std(nulls, axis=0)
    
    else:
        H0_mean = np.mean(attributions[label][1-label], axis=0)
        H0_std = np.std(attributions[label][1-label], axis=0)

    return {'alt': {'mean': H1_mean, 'std': H1_std}, 'null': {'mean': H0_mean, 'std': H0_std}}
