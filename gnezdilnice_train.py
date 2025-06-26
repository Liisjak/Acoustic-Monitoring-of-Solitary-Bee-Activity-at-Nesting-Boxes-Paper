import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import gc

import gnezdilnice_plotters
import gnezdilnice_models
import gnezdilnice_dataloader
import gnezdilnice_utils

class CELoss(torch.nn.Module):
    '''Cross-entropy loss that works with soft targets'''
    def __init__(self, num_classes):
        super(CELoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, logits, target_ohe):
        pred = F.log_softmax(logits, dim=1)
        ce = -torch.sum(pred * target_ohe, dim = 1)
        return ce.mean()
        
class step_counter_class():
    def __init__(self):
        self.count = 0
    def add(self):
        self.count +=1
        
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class performance_metrics_class():
    def __init__(self):
        self.dict = {'steps':[],
                     'epochs':[],
                     'times':[],
                     'train_loss':[],
                     'train_accuracy':[],
                     'test_loss':[],
                     'test_accuracy':[],
                     'test_specificity':[],
                     'test_sensitivity':[],
                     'test_precision':[],
                     'test_recall':[],
                     'test_f1':[],
                     'test_rocauc':[],
                        }  
    def add(self, string, value):
        self.dict[string].append(value)
        
def train_model(args, dataset, device):
    print(f'TRAINING MODEL {args.model}')
    print(f'\tDataset: {args.dataset}')
    print(f'\tMethod: {args.method}')
    print(f'\tSplit: {args.split}')
    print(f'\tInclude faint: {args.faint}')
    print(f'\tInclude high: {args.high}')
    print(f'\tNumber of epochs: {args.num_epochs}')
    print(f'\tBatch size: {args.batch_size}')
    print(f'\tMax learning rate: {args.lr_max}')
    print(f'\tUse learning rate scheduler: {args.use_sched}')
    print(f'\tOptimizer: Adam')
    print(f'\tDepth: {args.depth}')
    print(f'\tSeed(data): {args.seed_data}')

    # Fix the random seeds for reproduction purposes
    args.seed_fix = 4
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed_fix)  # CPU seed and all GPUs //Also affects dropout in the models
    #torch.cuda.manual_seed_all(seed_fix)  # GPU seed (_all for multi-GPU)
    random.seed(args.seed_fix)  # python seed
    np.random.seed(args.seed_fix) # numpy seed
    os.environ["PYTHONHASHSEED"] = str(args.seed_fix)
    
    RESULTS_ARGS = gnezdilnice_utils.check_folder(gnezdilnice_utils.results_dir(args))
    
    # Initialize train and test data loaders
    data_loader = gnezdilnice_dataloader.dataloader_gnezdilnice(args, dataset)
    train_loader = data_loader.run(mode='train')
    test_loader = data_loader.run(mode='test')
    
    # Print out the train and test data loaders properties
    print(f'\ttrain data size:')
    print(f'\t\twavs: {len(list(set(train_loader.dataset.wav_train)))}')
    print(f'\t\twavs: {sorted(list(set(train_loader.dataset.wav_train)))}')
    wavs_label = [0]*2
    wavs_flat = []
    segments_label = [0]*2
    for i, (wav, label) in enumerate(zip(train_loader.dataset.wav_train, train_loader.dataset.label_train)):
        segments_label[label] += 1
        if wav not in wavs_flat:
            wavs_label[label] += 1
            wavs_flat.append(wav)
    #for i, count in enumerate(wavs_label):
    #    print(f'\t\t\tlabel {i} :', count)
    print(f'\t\tsegments: {len(train_loader.dataset)}')
    for i, count in enumerate(segments_label):
        print(f'\t\t\tlabel {i} :', count)
    print(f'\ttest data size:')  
    print(f'\t\twavs: {len(list(set(test_loader.dataset.wav_test)))}')
    print(f'\t\twavs: {sorted(list(set(test_loader.dataset.wav_test)))}')
    wavs_label = [0]*2
    wavs_flat = []
    segments_label = [0]*2
    for i, (wav, label) in enumerate(zip(test_loader.dataset.wav_test, test_loader.dataset.label_test)):
        segments_label[label] += 1
        if wav not in wavs_flat:
            wavs_label[label] += 1
            wavs_flat.append(wav)
    #for i, count in enumerate(wavs_label):
    #    print(f'\t\t\tlabel {i} :', count)
    print(f'\t\tsegments: {len(test_loader.dataset)}')
    for i, count in enumerate(segments_label):
        print(f'\t\t\tlabel {i} :', count)

    # initialize the model
    torch.manual_seed(args.seed_fix) # set the initial weights with fixed seed
    if args.model == 'resnet9':
        model = gnezdilnice_models.ResNet9(num_classes = 2)
    model = nn.DataParallel(model) # sets to all available cuda devices
    model.to(device)      
    print(f'\tModel parameters count: {count_model_parameters(model)}')

    # calculate the number of steps
    args.num_steps = args.num_epochs*(len(train_loader.dataset)//args.batch_size)
    print(f'\tNumber of steps (calculated): {args.num_steps}')
    
    # initialize the criterion
    criterion = CELoss(num_classes=2)
    # initialize the optimizer
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr_max, weight_decay = args.weight_decay)
    # initialize the scheduler
    if args.use_sched:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer = optimizer, max_lr = args.lr_max, total_steps = args.num_steps) #COMMENT
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones=[int(args.num_steps*0.3),int(args.num_steps*0.8)], gamma=0.1)
    else: 
        scheduler = None #COMMENT
    # initialize the step counter
    step_counter = step_counter_class()
    # initialize performance metric tracker
    performance = performance_metrics_class()
    
    times = []
    lr_per_step = []
    step_saver = [] # used for variables that are saved every epoch
    epoch_plot = np.linspace(1, args.num_epochs, 11).astype('int') #51 #args.num_epochs+1
    epoch_plot = np.array(list(set(epoch_plot)))
    print(f'Number of plot epochs:', len(epoch_plot))
    print('->Training START<-')
    args.depth = 0
    for epoch in range(1, args.num_epochs + 1):
        time_start_epoch = time.time()
        
        # Train the model for one epoch
        loss_train_epoch, acc_train_epoch, lr_per_step_epoch = train_epoch(args, 
                                                                            model, 
                                                                            train_loader,
                                                                            device,
                                                                            optimizer, 
                                                                            scheduler, 
                                                                            criterion, 
                                                                            epoch, 
                                                                            step_counter)
        step_saver.append(step_counter.count)
        lr_per_step = lr_per_step + lr_per_step_epoch
        
        if epoch in epoch_plot:
            performance.add('epochs', epoch)
            performance.add('steps', step_counter.count)
            print(f'Plotting @ Epoch={epoch}, Step={step_counter.count}')
            # Train data accuracy
            performance.add('train_loss', loss_train_epoch)
            performance.add('train_accuracy', acc_train_epoch)
            ### Plots:
            if args.split == 'alltrain':
                # train and test accuracies vs epoch
                gnezdilnice_plotters.plot_train_test_acc(performance.dict['train_accuracy'], [0]*len(performance.dict['steps']), performance.dict['steps'], RESULTS_ARGS)
                # train and test losses vs epoch
                gnezdilnice_plotters.plot_train_test_loss(performance.dict['train_loss'], [0]*len(performance.dict['steps']), performance.dict['steps'], RESULTS_ARGS)
            else: 
                # Test data accuracy
                test_data_accuracy(args, model, test_loader, device, criterion, epoch, performance)
                print('\tMean test loss per epoch: {:.6f}'.format(performance.dict['test_loss'][-1]))
                # train and test accuracies vs epoch
                gnezdilnice_plotters.plot_train_test_acc(performance.dict['train_accuracy'], performance.dict['test_accuracy'], performance.dict['steps'], RESULTS_ARGS)
                # train and test losses vs epoch
                gnezdilnice_plotters.plot_train_test_loss(performance.dict['train_loss'], performance.dict['test_loss'], performance.dict['steps'], RESULTS_ARGS)
            # learning rate vs batch
            gnezdilnice_plotters.plot_lr_per_step(lr_per_step, RESULTS_ARGS, show = False)
            
        times.append(time.time()-time_start_epoch)
        if epoch in epoch_plot:
            # times vs epoch
            performance.add('times', np.sum(times))
            gnezdilnice_plotters.plot_times(times, step_saver, RESULTS_ARGS, show = False)
        # Save performance dict
        if epoch in epoch_plot:
            DICT = os.path.join(RESULTS_ARGS, f'performance.pkl')
            gnezdilnice_utils.save_dict(performance.dict, DICT)
    
    print('Finished Training')
    MODEL = os.path.join(RESULTS_ARGS, 'model.pth')
    torch.save(model.state_dict(), MODEL)
    
    # Release cuda memory (does this work???)
    for variable in [criterion, optimizer, scheduler, model]:
        del variable
    gc.collect() # Python thing
    with torch.no_grad():
        torch.cuda.empty_cache() # PyTorch thing
    return
    
def train_epoch(args, model, train_loader, device, optimizer, scheduler, criterion, epoch, step_counter):
    model.train()

    print(f'Step [{step_counter.count+1}/{args.num_steps}]')
    loss_per_batch = []
    lr_per_step = []
    pred_target_arr = [[], []]
    torch.manual_seed(args.seed_fix*635410 + step_counter.count) # this is here as some other methods (measure_test_accuracy()...) call into the pseudo-RNG and changes the order of training data of the next step
    if args.split == 'alltrain': # if all data is used for training, we have to shuffle based on seed_data, otherwise we always get the same result
        torch.manual_seed(args.seed_data*635410 + step_counter.count)
    for batch_idx, (data, target, buzztype, wav, segment, location) in enumerate(train_loader):
        data = data.to(device)
        target_ohe = F.one_hot(target, 2)
        target_ohe = target_ohe.to(device)
        
        # Mixup data (or not)
        #data, target_ohe, mix_indices, cut = mixup_data.mixup_data(args, data, target_ohe, frames, wav, step_counter, model, device)
        # Send through the model
        out = model(data)
        args.depth = 0 # reset in case we use 'latent-cutmix'
        # Measure loss
        loss = criterion(out, target_ohe)

        # Append loss
        loss_per_batch.append(loss.item())
        # Save predictions in dictionary
        pred = out.max(1, keepdim=True)[1]  # get the index of the max log-probability
        target = target_ohe.max(1, keepdim=True)[1] # reverse ohe
        for pred_i, target_i in zip(pred, target):
            pred_target_arr[0].append(pred_i.item())
            pred_target_arr[1].append(target_i.item())
        
        # Backward
        loss.backward()
        # Gradient clipping
        if args.grad_clip:
            nn.utils.clip_grad_value_(parameters = model.parameters(), clip_value = args.grad_clip)
            
        # Save some things
        learning_rate = optimizer.param_groups[0]['lr']
        lr_per_step.append(learning_rate)
        
        # Optimize
        torch.cuda.manual_seed_all(args.seed_fix) # SGD optimizer is non-deterministic, that's why we fix the seed
        optimizer.step()
        optimizer.zero_grad()
        if args.use_sched:
            scheduler.step() #COMMENT

        # Release cuda memory
        for var in [data, target_ohe, out, target]:
            var.cpu().detach()
            del var
        
        # Add a step to the counter
        step_counter.add()
        # Stop
        if not step_counter.count < args.num_steps:
            print(f'Training loop was stopped: epoch {epoch}, step {step_counter.count}')
            break
            
    acc_network = accuracy_score(pred_target_arr[0], pred_target_arr[1])*100
    return np.average(loss_per_batch), acc_network, lr_per_step
    
def test_data_accuracy(args, model, test_loader, device, criterion, epoch, performance):
    from collections import Counter

    model.eval()
   
    losses_all = 0
    preds_arr = []
    pred_probas_arr = []
    targets_arr = []
    with torch.no_grad():
        for data, target, _, _, _, _ in test_loader:
            data = data.to(device)
            target_ohe = F.one_hot(target, 2)
            target_ohe = target_ohe.to(device)
            size = data.shape[0]

            output = model(data)

            loss = criterion(output, target_ohe)
            losses_all += loss.item()*size   

            # calculate accuracy
            pred_proba = F.softmax(output, dim=1).cpu().detach().numpy()
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            target = target_ohe.max(1, keepdim=True)[1].cpu().detach().numpy() # reverse ohe
            for pred_i, pred_proba_i, target_i in zip(pred, pred_proba, target):
                preds_arr.append(pred_i.item())
                pred_probas_arr.append(pred_proba_i)
                targets_arr.append(target_i.item())
                
            # Release cuda memory
            for var in [data, target_ohe, output]:
                var.cpu().detach()
                del var

    # Accuracy
    acc_network = accuracy_score(preds_arr, targets_arr)*100
    performance.add('test_accuracy', acc_network)
    # Loss
    loss_total = losses_all/len(test_loader.dataset)
    performance.add('test_loss', loss_total)
    # Specificity and sensitiviy
    tn, fp, fn, tp = confusion_matrix(targets_arr, preds_arr).ravel()
    specificity = tn / (tn+fp)
    sensitivity = tp / (tp+fn)
    performance.add('test_specificity', specificity*100)
    performance.add('test_sensitivity', sensitivity*100)
    # Precision, recall, f1, aucroc
    f1 = f1_score(targets_arr, preds_arr)
    performance.add('test_f1', f1)
    precision = precision_score(targets_arr, preds_arr)
    performance.add('test_precision', precision)
    recall = recall_score(targets_arr, preds_arr)
    performance.add('test_recall', recall)
    rocauc = roc_auc_score(targets_arr, np.array(pred_probas_arr)[:, 1])
    performance.add('test_rocauc', rocauc)
    return 