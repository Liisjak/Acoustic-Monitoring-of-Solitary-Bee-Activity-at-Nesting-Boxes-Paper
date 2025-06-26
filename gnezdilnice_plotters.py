import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn import mixture
import scipy

import gnezdilnice_utils

def plot_epoch_loss(epoch_loss_corr, epoch_loss_incorr, epoch, RESULTS_ARGS, show = False):
    epoch_loss_all = np.append(epoch_loss_corr, epoch_loss_incorr)
    # normalize the losses
    epoch_loss_corr = epoch_loss_corr / np.max(epoch_loss_all)
    epoch_loss_incorr = epoch_loss_incorr / np.max(epoch_loss_all)
    bins = np.linspace(0, 1, 100)
    fig = plt.figure(figsize = (6, 6))
    plt.hist(epoch_loss_corr, bins, alpha=0.5, label='correct', color = 'royalblue')
    plt.hist(epoch_loss_incorr, bins, alpha=0.5, label='incorrect', color = 'crimson')
    plt.title(f'Epoch={epoch}')
    plt.xlabel('normalized loss')
    plt.ylabel('#samples')
    plt.legend()
    plt.grid()
    RESULTS_ARGS_LOSS = utils.check_folder(os.path.join(RESULTS_ARGS, 'losses'))
    FILENAME = os.path.join(RESULTS_ARGS_LOSS, f'epoch_loss_{epoch}.jpg')
    plt.savefig(FILENAME)
    if show:
        print(f'Figure saved to {FILENAME}')
        plt.show()
    else:
        plt.close()

def plot_train_test_acc(acc_train, acc_test, steps, RESULTS_ARGS):
    valid = False
    acc_test_max = np.max(acc_test)
    step_max = steps[acc_test.index(acc_test_max)]
    acc_test_max = np.round(acc_test_max, 2)
    acc_test_fin = np.round(acc_test[-1], 2)
    fig = plt.figure(figsize = (6, 6))
    valid_str = 'valid' if valid else 'test'
    valid_color = 'royalblue' if valid else 'forestgreen'
    plt.plot(steps, acc_train, label='train', color = 'darkorange')
    plt.plot(steps, acc_test, label = valid_str, color = valid_color)
    plt.axhline(y = acc_test_max, color = valid_color, linestyle = '--', label = f'{valid_str} max {acc_test_max} @step {step_max}')
    plt.axhline(y = acc_test_fin, color = valid_color, linestyle = '-.', label = f'{valid_str} final {acc_test_fin}')
    plt.ylim(bottom = 0, top = 110)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy [%]')
    plt.legend()
    plt.grid()
    #acc_dict = {'train':acc_train, 'test':acc_test, 'steps':steps}
    #DICT = os.path.join(RESULTS_ARGS, f'accuracy.pkl')
    #utils.save_dict(acc_dict, DICT)
    PLOT = os.path.join(RESULTS_ARGS, f'accuracy.jpg')
    plt.savefig(PLOT)
    plt.close()

def plot_train_test_loss(loss_train, loss_test, steps, RESULTS_ARGS):
    valid = False
    fig = plt.figure(figsize = (6, 6))
    valid_str = 'valid' if valid else 'test'
    valid_color = 'royalblue' if valid else 'forestgreen'
    train_final = np.round(loss_train[-1], 2)
    valid_final = np.round(loss_test[-1], 2)
    plt.plot(steps, loss_train, label='train', color = 'darkorange')
    plt.axhline(y = train_final, color = 'darkorange', linestyle = '-.', label = f'train final {train_final}')
    plt.plot(steps, loss_test, label = valid_str, color = valid_color)
    plt.axhline(y = valid_final, color = valid_color, linestyle = '-.', label = f'{valid_str} final {valid_final}')
    #plt.ylim(bottom = 0, top = 110)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    FILENAME = os.path.join(RESULTS_ARGS, f'loss.jpg')
    plt.savefig(FILENAME)
    plt.close()
        
def plot_times(times, steps, RESULTS_ARGS, show = False):
    times_sum = np.sum(times)
    hours, rem = divmod(times_sum, 3600)
    minutes, seconds = divmod(rem, 60)
        
    fig = plt.figure(figsize = (6, 6))
    plt.plot(steps, times, label = r'times', color = 'k')
    plt.ylim(bottom = 0)
    plt.xlabel('Steps')
    plt.ylabel(r'times [s]')
    plt.title("Total {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    plt.legend()
    plt.grid()
    FILENAME = os.path.join(RESULTS_ARGS, f'times.jpg')
    plt.savefig(FILENAME)
    if show:
        print(f'Figure saved to {FILENAME}')
        plt.show()
    else:
        plt.close()
                
def plot_lr_per_step(lr_per_step, RESULTS_ARGS, show = False):
    num_steps = len(lr_per_step)
        
    fig = plt.figure(figsize = (6, 6))
    plt.plot(np.arange(1, num_steps+1, 1), lr_per_step, label = f'learning_rate', color = 'k')
    plt.ylim(bottom = 0)
    plt.xlabel('Step')
    plt.ylabel(r'Learning rate')
    plt.legend()
    plt.grid()
    FILENAME = os.path.join(RESULTS_ARGS, f'learning_rate.jpg')
    plt.savefig(FILENAME)
    if show:
        print(f'Figure saved to {FILENAME}')
        plt.show()
    else:
        plt.close()