import torch
import torchvision
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def saliency_map(data, target_ohe, model, device, with_respect_to):
    size = data.shape[0]
    target = target_ohe.max(1, keepdim=True)[1] # reverse ohe
    data.requires_grad_()
    # Forward pass data
    model.eval()
    out = model(data)
    # returns scores of the correct class and puts them all into 1d tensor
    if with_respect_to == 'correct-class':
        label_view = target.view(-1, 1) # for correct class
    elif with_respect_to == 'buzz-class':
        selected_label = 1
        label_view = (torch.ones((size, 1)).type(torch.int64)*selected_label).to(device)
        print('yes')
    else: 
        raise Exception("No label view set") 
    scores = (out.gather(1, label_view).squeeze())  # print(scores.shape[0]) # 500
    # Calculate gradients
    scores.backward(torch.FloatTensor([1.0]*scores.shape[0]).to(device))
    # Get saliency map
    saliency = data.grad.data.abs() 
    # Apply the GaussianBlur transform to the batch of images
    saliency = torch.stack([torch.from_numpy(gaussian_filter(img.detach().cpu().numpy(), sigma=0.8)).to(device) for img in saliency])
    '''
    # Batch-level normalization: normalize between 0 and 1 
    saliency_min = torch.min(saliency)
    saliency_max = torch.max(saliency)
    saliency -= saliency_min
    saliency /= saliency_max
    '''
    # Instance-level normalization: normalize between 0 and 1 
    saliency_size = saliency.size()
    saliency = saliency.view(saliency.size(0), -1)
    saliency -= saliency.min(1, keepdim=True)[0]
    saliency /= saliency.max(1, keepdim=True)[0]
    saliency = saliency.view(saliency_size)
    # Fill the missing values as some saliencies that were full-zero are now "nan" after normalization
    saliency = torch.nan_to_num(saliency, nan=0.0)
    return saliency
    
def visualize_spec_saliency(data, sal, target, fmin, fmax):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize = (8, 3), gridspec_kw={'wspace':-0.3})
    # spectrogram
    #axes[0].imshow(np.squeeze(data.detach().cpu().numpy()), origin='lower', vmin=-4.2, vmax=3.14)
    axes[0].imshow(np.squeeze(data.detach().cpu().numpy()), origin='lower', vmin=-3, vmax=3)
    axes[0].imshow(np.squeeze(data.detach().cpu().numpy()), origin='lower')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_xlim((0, 128))
    axes[0].set_xticks([0, 32, 64, 96, 128])
    axes[0].set_xticklabels([0, 1, 2, 3, 4])
    axes[0].set_ylabel('Frequency [Hz]')
    axes[0].set_ylim((0, 128))
    axes[0].set_yticks([0, 32, 64, 96, 128])
    axes[0].set_yticklabels([int(x) for x in np.linspace(fmin, fmax, 5)])
    axes[0].set_title('spectrogram')
    # saliency
    axes[1].set_xlabel('Time [s]')
    axes[1].set_xlim((0, 128))
    axes[1].set_xticks([0, 32, 64, 96, 128])
    axes[1].set_xticklabels([0, 1, 2, 3, 4])
    axes[1].set_ylabel('Frequency [Hz]')
    axes[1].set_ylim((0, 128))
    axes[1].set_yticks([0, 32, 64, 96, 128])
    axes[1].set_yticklabels([int(x) for x in np.linspace(fmin, fmax, 5)])
    axes[1].imshow(np.squeeze(sal.detach().cpu().numpy()), origin='lower', cmap='magma', vmin=0, vmax=0.35)
    #axes[1].imshow(np.squeeze(sal.detach().cpu().numpy()), origin='lower', cmap='magma', vmin=0, vmax=0.2)
    #axes[1].imshow(np.squeeze(sal.detach().cpu().numpy()), origin='lower', cmap='magma')
    axes[1].set_title('saliency')
    for ax in fig.get_axes():
        ax.label_outer()
    fig.suptitle(f'Class: {target.item()}')
    plt.show()
    plt.close()
    return

    
    
    