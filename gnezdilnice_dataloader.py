import os
import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import random

import gnezdilnice_utils

class dataset_gnezdilnice(Dataset):
    def __init__(self, dataset, split, method, faint_bool, high_bool, mode, seed_data, transform):
        self.dataset = dataset
        self.split = split
        self.method = method
        self.faint_bool = faint_bool
        self.high_bool = high_bool
        self.mode = mode
        self.seed_data = seed_data
        self.transform = transform
        
        self.data = np.array(self.dataset['data'])
        self.label = np.array(self.dataset['label'])
        self.buzztype = np.array(self.dataset['buzz_type'])
        self.wav = np.array(self.dataset['wav'])
        self.segment = np.array(self.dataset['segment'])
        self.location = np.array(self.dataset['location'])
        # to prevent TypeError
        self.buzztype[self.buzztype==None] = ''
                
        if not self.faint_bool:
            # Remove faint buzzes
            indices = [i for i, d in enumerate(self.buzztype) if d not in ['faint', 'faint_long']]
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.buzztype = self.buzztype[indices]
            self.wav = self.wav[indices]
            self.segment = self.segment[indices]
            self.location = self.location[indices]    
        if not self.high_bool:
            # Remove faint buzzes
            indices = [i for i, d in enumerate(self.buzztype) if d not in ['high', 'high_long']]
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.buzztype = self.buzztype[indices]
            self.wav = self.wav[indices]
            self.segment = self.segment[indices]
            self.location = self.location[indices]
            
        if self.split == 'alltrain':
            # select train data (all)
            self.data_train = self.data
            self.label_train = self.label
            self.buzztype_train = self.buzztype
            self.wav_train = self.wav
            self.segment_train = self.segment
            self.location_train = self.location
            # select test data (empty)
            self.data_test = np.array([])
            self.label_test = np.array([])
            self.buzztype_test = np.array([])
            self.wav_test = np.array([])
            self.segment_test = np.array([])
            self.location_test = np.array([])
        elif self.split == 'loc':
            if self.seed_data not in [1, 2, 3]:
                raise Exception(f'"seed_data" ({self.seed_data}) must be either 1, 2, or 3')
            cv_folds = {}
            locations = list(set(self.location))
            for f, loc in enumerate(locations):
                train_indices = [i for i, l in enumerate(self.location) if l!=loc]
                test_indices = [i for i, l in enumerate(self.location) if l==loc]
                cv_folds[f] = {'train':train_indices, 'test':test_indices}
            train_indices_sd = cv_folds[self.seed_data-1]['train']
            test_indices_sd = cv_folds[self.seed_data-1]['test']
            # select train data
            self.data_train = self.data[train_indices_sd]
            self.label_train = self.label[train_indices_sd]
            self.buzztype_train = self.buzztype[train_indices_sd]
            self.wav_train = self.wav[train_indices_sd]
            self.segment_train = self.segment[train_indices_sd]
            self.location_train = self.location[train_indices_sd]
            # select test data
            self.data_test = self.data[test_indices_sd]
            self.label_test = self.label[test_indices_sd]
            self.buzztype_test = self.buzztype[test_indices_sd]
            self.wav_test = self.wav[test_indices_sd]
            self.segment_test = self.segment[test_indices_sd]
            self.location_test = self.location[test_indices_sd]
        elif self.split == 'cv':
            number_of_folds = 5
            if self.seed_data not in np.arange(1, number_of_folds+1, 1):
                raise Exception(f'Parameter "self.seed" ({self.seed_data}) must be in {list(np.arange(1, number_of_folds+1, 1))}')
            # split the wavs into 6 categories: 2 classes per location
            segs = {'staritrgobkolpi_0': [], 'staritrgobkolpi_1': [], 'prelesje_0': [], 'prelesje_1': [], 'radenci_0': [], 'radenci_1': []}
            for i, (label_i, location_i) in enumerate(zip(self.label, self.location)):
                segs[f'{location_i}_{label_i}'].append(i)
            cv_folds = {}
            for f in range(number_of_folds):
                test_indices = []
                train_indices = []
                for key in segs:
                    # this manoeuver does not let the overlapping segments be in separate sets
                    arr = segs[key]
                    num_test_segs_key = np.round(len(arr)/number_of_folds)
                    fold_starts = [int(num_test_segs_key*k) for k in range(number_of_folds)]
                    fold_starts = fold_starts + [len(arr)]
                    if f == 0:
                        test_indices_key = arr[0:fold_starts[1]]
                        train_indices_key = arr[fold_starts[1]+1:]
                    elif 0<f<number_of_folds-1:
                        test_indices_key = arr[fold_starts[f]:fold_starts[f+1]]
                        train_indices_key = arr[:fold_starts[f]-1] + arr[fold_starts[f+1]+1:]
                    else:
                        test_indices_key = arr[fold_starts[f]:]
                        train_indices_key = arr[:fold_starts[f]-1]
                    test_indices = test_indices + test_indices_key
                    train_indices = train_indices + train_indices_key
                cv_folds[f] = {'train':train_indices, 'test':test_indices}
            train_indices_sd = cv_folds[self.seed_data-1]['train']
            test_indices_sd = cv_folds[self.seed_data-1]['test']
            # select train data
            self.data_train = self.data[train_indices_sd]
            self.label_train = self.label[train_indices_sd]
            self.buzztype_train = self.buzztype[train_indices_sd]
            self.wav_train = self.wav[train_indices_sd]
            self.segment_train = self.segment[train_indices_sd]
            self.location_train = self.location[train_indices_sd]
            # select test data
            self.data_test = self.data[test_indices_sd]
            self.label_test = self.label[test_indices_sd]
            self.buzztype_test = self.buzztype[test_indices_sd]
            self.wav_test = self.wav[test_indices_sd]
            self.segment_test = self.segment[test_indices_sd]
            self.location_test = self.location[test_indices_sd]
        else:
            raise Exception('Split must be either leave-one-location-out ("loc") or cross-validation ("cv")')
        
    def __getitem__(self, index):
        if self.mode == 'train':
            spectrogram, target, buzztype, wav, segment, location = self.data_train[index], self.label_train[index], self.buzztype_train[index], self.wav_train[index], self.segment_train[index], self.location_train[index]
            spectrogram = torch.tensor(spectrogram)
            spectrogram = spectrogram[None, :] # add channel dimensions
            return spectrogram, target, buzztype, wav, segment, location
        elif self.mode == 'test':
            spectrogram, target, buzztype, wav, segment, location = self.data_test[index], self.label_test[index], self.buzztype_test[index], self.wav_test[index], self.segment_test[index], self.location_test[index]
            spectrogram = torch.tensor(spectrogram)
            spectrogram = spectrogram[None, :] # add channel dimensions
            return spectrogram, target, buzztype, wav, segment, location

    def __len__(self):
        if self.mode == 'test':
            return len(self.data_test)
        elif self.mode == 'train':
            return len(self.data_train)

class dataloader_gnezdilnice():
    def __init__(self, args, dataset):
        self.dataset = dataset
        self.dataset_name = args.dataset
        self.method = args.method
        self.split = args.split
        self.faint_bool = args.faint
        self.high_bool = args.high
        self.seed_data = args.seed_data
        self.batch_size = args.batch_size  
        self.seed_fix = args.seed_fix
        
        #DATASET = os.path.join(args.DATA, f'{args.dataset}.dat')
        #self.dataset = gnezdilnice_utils.file2dict(DATASET)
        
        if self.method in ['flip', 'cuttime', 'cutfreq']:
            self.transform_train = None
        else:
            self.transform_train = None

    def run(self, mode):
        if mode == 'train':
            train_dataset = dataset_gnezdilnice(dataset = self.dataset,
                                           split = self.split,
                                           method = self.method,
                                           faint_bool = self.faint_bool,
                                           high_bool = self.high_bool,
                                           mode = 'train',
                                           seed_data = self.seed_data,
                                           transform = self.transform_train,
                                              )                       
                                              
            random.seed(self.seed_fix)
            torch.manual_seed(self.seed_fix)
            train_loader = DataLoader(dataset=train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      #generator = torch.Generator(device='cpu'), # this line fixes the shuffle random when iterating
                                      drop_last = True)                        
            return train_loader
        elif mode == 'test':
            test_dataset = dataset_gnezdilnice(dataset = self.dataset,
                                          split = self.split,
                                          method = self.method,
                                          faint_bool = self.faint_bool,
                                          high_bool = self.high_bool,
                                          mode = 'test',
                                          seed_data = self.seed_data,
                                          transform = None,
                                              )  
        
            test_loader = DataLoader(dataset=test_dataset,
                                     batch_size=128, # manually set to higher value to make it go faster
                                     shuffle=False,
                                     drop_last = False)                                                         
            return test_loader


    

