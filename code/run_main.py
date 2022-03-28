import os, sys
sys.path.insert(0, "../")
import torch
from torch import nn
import pytorch_lightning as pl
from argparse import ArgumentParser
import numpy as np 
from tqdm import tqdm
from classifier import AvtDatamodule
from tester import massiv_test
from torch.utils.data import DataLoader, random_split
import pdb
torch.set_default_tensor_type(torch.DoubleTensor)
from dataloader import AvtDataLoader, collate_fn
from classifier import AvtClassifier    

def main(args):
    
    pl.utilities.seed.seed_everything(seed=42)    
    model = AvtClassifier(args)
    data_module = AvtDatamodule(args)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc1', verbose=True, save_top_k=1, mode='max')
    trainer = pl.Trainer(gpus=-1, max_epochs=args.max_epochs, callbacks= [checkpoint_callback])
    trainer.fit(model, data_module)

def test_model(args):
    
    test_data = AvtDataLoader(args, args.test_file, split = "test")
        
    test_dataloader = DataLoader(test_data, batch_size = args.batch_size, collate_fn = collate_fn, num_workers = args.num_workers)
    print("Processing test data ==> {}".format(len(test_data)))

    model_path = args.model_path
    print("Testing model ==> {}".format(model_path))
    model = AvtClassifier(args).load_from_checkpoint(model_path)
    model.cuda()
    massiv_test(args, model, test_dataloader)

if __name__ == '__main__':
    
    parser = ArgumentParser()
    parser.add_argument('--run', type=str, default='00')
    parser = AvtClassifier.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    # Unimodal and multimodal experiments
    parser.add_argument('--mode', type=str, default="vs", choices=["vs", "vs_as", "vs_2as"])
    parser.add_argument('--train_file', type=str, default="")
    parser.add_argument('--pred_loc', type=str, default=None)
    parser.add_argument('--test_file', type=str, default="")
    parser.add_argument('--val_file', type=str, default="")
    parser.add_argument('--map_file_location', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=30)
    parser.add_argument('--num_labels', type=int, default=34, help="number of concept categories")
    parser.add_argument('--video_location', type=str, default=None, help="path to the video features")
    parser.add_argument('--audio_location', type=str, default=None, help="path to the audio features")
    parser.add_argument('--audio_featloc_second', type=str, default=None, help="path to the second audio features")
    parser.add_argument('--phase', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--num_classes', type=int, default=34, help="number of categories")
    # Language is set to "all" for experiments over complete data
    # Source and Target language can be set for cross lingual zero shot experiments
    parser.add_argument('--src_lang', type=str, default="all")
    parser.add_argument('--tgt_lang', type=str, default="all")
    # Path to the pretrained model for testing phase
    parser.add_argument('--model_path', type=str, default="")
    args = parser.parse_args()
    
    if args.phase == "train":
        main(args)
    else:
        test_model(args)
    



