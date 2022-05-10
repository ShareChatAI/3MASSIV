import os, sys
sys.path.insert(0, "../")
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from dataloader import MassivDataset, collate_fn
from torchvision import datasets, transforms
from argparse import ArgumentParser
import numpy as np 
from tqdm import tqdm
import torchmetrics
import pdb
from torch.autograd import Variable

torch.set_default_tensor_type(torch.DoubleTensor)

class Accuracy_topk(torchmetrics.metric.Metric):
    def __init__(self, topk, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.k = topk
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        assert preds.shape[0] == target.shape[0]
        _, pred = preds.topk(self.k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        self.correct += correct.reshape(-1).float().sum(0) 
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total

class MassivDatamodule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        
    def setup(self, stage=None):

        self.train_data = MassivDataset(self.args, self.args.train_file, split = "train")
        self.val_data = MassivDataset(self.args, self.args.val_file, split = "val")
        
        print("Number of training samples ==> {}".format(len(self.train_data)))
        print("Number of validation samples ==> {}".format(len(self.val_data)))
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.args.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.args.batch_size, collate_fn=collate_fn, num_workers=self.args.num_workers)

class MassivClassifier(pl.LightningModule):

    def __init__(self, args):

        super().__init__()
        self.args = args
        self.lr = args.lr
        self.save_pred = args.save_pred
        self.save_hyperparameters()
        self.mode = args.mode
        
        if self.mode in ["vs_as"]:
            p1 = 0.5
            audio_size = 512 # CLSRIL features are 512 dimensional
            #audio_size = 128 # VGG features are 128 dimensional
            self.model_video = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), 
            nn.Linear(1024, 512), nn.Softmax()) 
            self.model_audio = nn.Sequential(nn.Linear(audio_size, 512), nn.BatchNorm1d(512), nn.ReLU(), 
            nn.Linear(512, 512), nn.Softmax()) 
            self.model_fusion = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p1), 
            nn.Linear(512, args.num_classes))
            print(self.model_video)
            print(self.model_audio)
            print(self.model_fusion)
        
        elif self.mode in ["vs_2as_ue"]:
            p1 = 0.5
            audio_size = 640 # CLSRIL + VGG (512 + 128)
            self.model_video = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), 
            nn.Linear(1024, 512), nn.Softmax()) 
            self.model_audio = nn.Sequential(nn.Linear(audio_size, 512), nn.BatchNorm1d(512), nn.ReLU(), 
            nn.Linear(512, 512), nn.Softmax()) 
            self.model_creator = nn.Sequential(nn.Linear(args.num_classes, 512), nn.BatchNorm1d(512), nn.ReLU(), 
            nn.Linear(512, 512), nn.Softmax()) 
            self.model_fusion = nn.Sequential(nn.Linear(1536, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p1), 
            nn.Linear(512, args.num_classes))
            print(self.model_video)
            print(self.model_audio)
            print(self.model_creator)
            print(self.model_fusion)
        
        elif self.mode in ["vs_2as"]:
            audio_size = 640 # CLSRIL + VGG (512 + 128)
            p1 = 0.5
            self.model_video = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.ReLU(), 
            nn.Linear(1024, 512), nn.Softmax()) 
            self.model_audio = nn.Sequential(nn.Linear(audio_size, 512), nn.BatchNorm1d(512), nn.ReLU(), 
            nn.Linear(512, 512), nn.Softmax()) 
            self.model_fusion = nn.Sequential(nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p1), 
            nn.Linear(512, args.num_classes))
            print(self.model_video)
            print(self.model_audio)
            print(self.model_fusion)
        
        elif self.mode in ["vs"]:
            p1 = 0.5
            self.model_video = nn.Sequential(nn.Linear(2048, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p1), 
            nn.Linear(512, args.num_classes)) 
            print(self.model_video)
        
        elif self.mode in ["as"]:
            p1 = 0.5
            audio_size = 512 # CLSRIL features are 512 dimensional
            #audio_size = 128 #  VGG features are 128 dimensional
            self.model_audio = nn.Sequential(nn.Linear(audio_size, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p1), 
            nn.Linear(512, args.num_classes)) 
            print(self.model_audio)
        
        elif self.mode in ["2as"]:
            p1 = 0.5
            audio_size = 640 # CLSRIL + VGG
            self.model_audio = nn.Sequential(nn.Linear(audio_size, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(p1), 
            nn.Linear(512, args.num_classes)) 
            print(self.model_audio)
            
        else:
            print("Mode Not Supported !!")
            sys.exit()
        
        self.loss = torch.nn.CrossEntropyLoss()
        self.level1_train_accuracy = Accuracy_topk(topk=1)
        self.level1_train_accuracy3 = Accuracy_topk(topk=3)
        self.level1_train_accuracy5 = Accuracy_topk(topk=5)
        self.level1_test_accuracy = Accuracy_topk(topk=1)
        self.level1_test_accuracy3 = Accuracy_topk(topk=3)
        self.level1_test_accuracy5 = Accuracy_topk(topk=5)
        self.level1_val_accuracy = Accuracy_topk(topk=1)
        self.level1_val_accuracy3 = Accuracy_topk(topk=3)
        self.level1_val_accuracy5 = Accuracy_topk(topk=5)
    
    def forward(self, x):
        
        if self.mode in ["vs"]:
            out_video = self.model_video(x[:,:2048])
            return out_video
        
        elif self.mode in ["as", "2as"]:
            out_video = self.model_audio(x)
            return out_video
        
        elif self.mode in ["vs_as", "vs_2as"]:
            out_video = self.model_video(x[:,:2048])
            out_audio = self.model_audio(x[:,2048:])
            out = torch.cat([out_video, out_audio], axis=1)
            return self.model_fusion(out)
        
        elif self.mode in ["vs_2as_ue"]:
            out_video = self.model_video(x[:,:2048])
            out_audio = self.model_audio(x[:,2048:2688])
            out_creator = self.model_creator(x[:,2688:])
            out = torch.cat([out_video, out_audio, out_creator], axis=1)
            return self.model_fusion(out)

        else:
            print(f"Mode {self.mode} not supported !!")
            sys.exit()
    
    def training_step(self, train_batch, batch_idx):
           
        x = train_batch['data']
        y = train_batch['label']
        logits = self.forward(x)

        loss = self.loss(logits, y)
        self.level1_train_accuracy(logits, y)
        self.level1_train_accuracy3(logits, y)
        self.level1_train_accuracy5(logits, y)

        self.log('tr_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('tr_acc1', self.level1_train_accuracy.compute(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('tr_acc3', self.level1_train_accuracy3.compute(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('tr_acc5', self.level1_train_accuracy5.compute(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        
        return {'loss': loss}
        
    def validation_step(self, val_batch, batch_idx):
        x = val_batch['data']
        y = val_batch['label']
        logits = self.forward(x)
        loss = self.loss(logits, y)

        self.level1_val_accuracy(logits, y)
        self.level1_val_accuracy3(logits, y)
        self.level1_val_accuracy5(logits, y)
        
        self.log('val_loss', loss, on_step=True, on_epoch=True,  logger=True)
        self.log('val_acc1', self.level1_val_accuracy.compute(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc3', self.level1_val_accuracy3.compute(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc5', self.level1_val_accuracy5.compute(), on_step=False, on_epoch=True, logger=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc1' : self.level1_val_accuracy.compute(), 'val_acc3' : self.level1_val_accuracy3.compute(), 'val_acc5' : self.level1_val_accuracy5.compute()}
 
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input', type=int, default=-1)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--save_pred', type=str, default=None)
        return parser

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.3) 
        return [optimizer], [scheduler]
