import numpy as np
from tqdm import tqdm
import copy

import torch

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
plt.rcParamsDefault['font.family'] = "sans-serif"
plt.rcParamsDefault['font.sans-serif'] = "Arial"
plt.rcParams['font.size'] = 14
plt.rcParams["errorbar.capsize"] = 0.5

from sklearn import metrics

def train(model, args, X, y, optimizer, criterion, permutation):
    epoch_losses = []
    epoch_accs = []

    model.train()

    for i in range(0, X.size()[0], args.batch_size):

        optimizer.zero_grad()

        indices = permutation[i:i + args.batch_size]
        batch_x, batch_y = X[indices], y[indices]

        y_pred = model(batch_x,)
        loss = criterion(
            y_pred.view(-1, args.out_dim), 
            batch_y.view(-1)
        )

        acc = model.accuracy(y_pred, batch_y)

        loss.backward()
        optimizer.step()

        epoch_losses += [loss.item()]
        epoch_accs += [acc.item()]

    return  np.sum(epoch_losses) / len(epoch_losses), np.sum(epoch_accs) / len(epoch_accs)


def evaluate(model, args, X, y, criterion, permutation):
    epoch_losses = []
    epoch_accs = []

    model.eval()

    with torch.no_grad():
        for i in range(0, X.size()[0], args.batch_size):
            indices = permutation[i:i + args.batch_size]
            batch_x, batch_y = X[indices], y[indices]

            y_pred = model(batch_x,)
            loss = criterion(
                y_pred.view(-1, args.out_dim), 
                batch_y.view(-1)
            )

            acc = model.accuracy(y_pred, batch_y)

            epoch_losses += [loss.item()]
            epoch_accs += [acc.item()]
    
    return np.sum(epoch_losses) / len(epoch_losses), np.sum(epoch_accs) / len(epoch_accs)

def fit(model, args, data, permuts, criterion, optimizer, show_progress=False):
    X_tr, y_tr, X_val, y_val, X_te, y_te = data
    permut_train, permut_valid, permut_test = permuts
    best_valid_loss = float('inf')

    train_loss, train_acc = np.zeros(args.num_epochs), np.zeros(args.num_epochs)
    valid_loss, valid_acc = np.zeros(args.num_epochs), np.zeros(args.num_epochs)

    for epoch in (range(args.num_epochs)):
        train_loss[epoch], train_acc[epoch] = train(
            model, args, X_tr, y_tr, 
            optimizer, criterion, permut_train
        )
        valid_loss[epoch], valid_acc[epoch] = evaluate(
            model, args, X_val, y_val, 
            criterion, permut_valid
        )

        if valid_loss[epoch] < best_valid_loss:
            best_valid_loss = valid_loss[epoch]
            # torch.save(model.state_dict(), model_file)
            best_model = copy.deepcopy(model)
            best_epoch = epoch
        
        if show_progress:
            print(f'Epoch: {epoch+1:02}')
            print(f'\tTrain Loss: {train_loss[epoch]:.3f} | Train Acc: {train_acc[epoch]*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss[epoch]:.3f} |  Val. Acc: {valid_acc[epoch]*100:.2f}%')

    history = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'valid_loss': valid_loss,
        'valid_acc': valid_acc
    }

    # torch.save(model.state_dict(), model_file)
    # with open(history_file, 'wb') as f:
    #     pickle.dump(history, f)

    return {'best_model':best_model, 'history':history, 'best_epoch':best_epoch}

def plot_training_history(args, fit_outputs, save_fig=False, fig_file=None):
    histories = fit_outputs['histories']
    best_epochs = fit_outputs['best_epochs']

    fig, axs = plt.subplots(
        nrows=2, ncols=1,
        figsize=(11,5),
        dpi=150
    )

    plt.subplots_adjust(
        left=None, bottom=None, 
        right=None, top=None, 
        wspace=None, hspace=0.5
    )

    colors = {
        'train_loss':'tomato', 'train_acc': 'tomato', 
        'valid_loss':'forestgreen', 'valid_acc':'forestgreen',
    }

    ax = axs[0] # loss
    for k in [k for k in histories.keys() if 'loss' in k]:
        mean_ts = np.mean(histories[k], axis=0)
        std_ts = np.std(histories[k], axis=0)

        ax.plot(mean_ts, color=colors[k], label=k)

        ax.fill_between(
            np.arange(args.num_epochs),
            (mean_ts - std_ts),
            (mean_ts + std_ts),
            color=colors[k],
            alpha=0.3,
        )
    ax.set_ylabel(f"losses")
    ax.set_xlabel(f"epochs")
    ax.set_title(f"noise level {args.noise_level}")
    ax.legend()
    ax.grid(True)

    ax = axs[1] # acc
    for k in [k for k in histories.keys() if 'acc' in k]:
        mean_ts = np.mean(histories[k], axis=0)
        std_ts = np.std(histories[k], axis=0)

        ax.plot(mean_ts, color=colors[k], label=k)

        ax.fill_between(
            np.arange(args.num_epochs),
            (mean_ts - std_ts),
            (mean_ts + std_ts),
            color=colors[k],
            alpha=0.3,
        )
    
    # best epochs
    min_ep, max_ep = min(best_epochs), max(best_epochs)
    ax.axvspan(min_ep, max_ep, alpha=0.3, color='blue', label='best_epochs')

    ax.set_ylabel(f"accuracies")
    ax.set_xlabel(f"epochs")
    ax.set_title(f"valid acc {np.max(mean_ts):.3f}")
    ax.legend()
    ax.grid(True)

    # save
    if save_fig:
        fig.savefig(
            fig_file,
            dpi=150,
            format='png',
            bbox_inches='tight',
            transparent=False
        )

def plot_confusion_matrix(args, labels, pred_labels):
    fig = plt.figure(figsize=(3*args.num_classes, 3*args.num_classes))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.confusion_matrix(labels, pred_labels)
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(args.num_classes))
    cm.plot(values_format='d', cmap='Blues', ax=ax)