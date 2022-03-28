import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import os
import json
import glob, pickle
torch.set_default_tensor_type(torch.DoubleTensor)
import pdb 
from tqdm import tqdm
from numpy.linalg import norm
import glob

class AvtDataLoader(Dataset):
    
    def __init__(self, args, csv, split):
        self.args = args
        print(f"Reading {csv}")
        self.csv = pd.read_csv(csv)
        self.split = split
        self.target_ids = []
        self.post_ids = []
        self.video_feats_paths = args.video_location
        self.audio_feats_paths = args.audio_location
        self.audio_feats_paths_second = args.audio_featloc_second
        self.mode = args.mode
        self.num_classes = args.num_classes
        self.__update_data__()
    
    def __update_data__(self):
        for idx in tqdm(range(len(self.csv))):
            path_str = self.csv['concept'].iloc[idx]
            tgt_id = self.csv['concept_idx'].iloc[idx]
            self.post_ids.append(self.csv['post_idx'].iloc[idx])
            self.target_ids.append(tgt_id)

    def __len__(self):
        return len(self.post_ids)

    def __get_resnext_feats__(self, postId):
        np_path = os.path.join(self.video_feats_paths, str(postId))
        if os.path.isdir(np_path):
            np_path = np_path
        else:
            return None
        all_arrs = []
        all_fts = glob.glob(np_path+"/feats_*.npy")
        if len(all_fts) == 0:
            return None
        for ft in all_fts:
            arr = np.load(ft, allow_pickle=True)
            all_arrs.append(arr.squeeze())
        if len(all_arrs)==1:
            arr = all_arrs[0]
        else:
            arr = np.mean(all_arrs, axis=0)
        arr = arr.astype('double')
        return arr
        
    def __get_audio_feats__(self, postId, audio_feats_paths):

        np_audio_path = os.path.join(audio_feats_paths, str(postId) + '/audio_feats.npy')
        if not os.path.isfile(np_audio_path):
            return None
    
        aud = np.load(np_audio_path, allow_pickle=True)
        if len(aud.shape)==0:
            return None
        else:
            aud = aud.squeeze()
            if len(aud.shape)>1:
                # CLSRIL - averaging over temporal dimensions
                aud = np.mean(aud, axis=1)
            else:
                # VGGISH
                return aud
        return aud
    

    def __getitem__(self, idx):

        postId = self.post_ids[idx]
        path_int = self.target_ids[idx]
        
        arr = self.__get_resnext_feats__(postId)
        aud = self.__get_audio_feats__(postId, self.audio_feats_paths)
        
        if aud is None:
            return None
        if arr is None:
            return None
        
        arr = arr/norm(arr)
        aud = aud/norm(aud)

        if self.mode in ["vs_as"]:
            arr = np.concatenate([arr.reshape(-1),  aud.astype('double')])
        
        elif self.mode in ["vs_2as"]:
            vgg_audio_feats_paths = self.args.audio_featloc_second
            aud_vgg = self.__get_audio_feats__(postId, vgg_audio_feats_paths)
            if aud_vgg is None:
                return None
            aud_vgg = aud_vgg/norm(aud_vgg)
            arr = np.concatenate([arr.reshape(-1),  aud.astype('double'), aud_vgg.astype('double')])
        
        elif self.mode == "vs":
            arr = arr.reshape(-1)
        
        elif self.mode == "as":
            arr = aud.astype('double').reshape(-1)
        
        elif self.mode == "2as":
            vgg_audio_feats_paths = self.audio_feats_paths_second
            aud_vgg = self.__get_audio_feats__(postId, vgg_audio_feats_paths)
            if aud_vgg is None:
                return None
            aud_vgg = aud_vgg/norm(aud_vgg)
            arr = np.concatenate([aud.astype('double').reshape(-1), aud_vgg.astype('double')])
        
        return {'data' : torch.from_numpy(arr), 'label' : path_int, 'id' : postId}

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
