from tqdm import tqdm
import torch
import os
import numpy as np
import pdb 
from PIL import Image
import torch.nn as nn

def massiv_test(args, model, test_dataloader):
    
    num_labels = args.num_classes

    model.eval()
    test_acc_top1 = 0.0
    test_acc_top3 = 0.0
    test_acc_top5 = 0.0

    cm = np.zeros((num_labels, num_labels))
    
    total = 0

    for i, batch in enumerate(tqdm(test_dataloader)):
        
        data = batch["data"].cuda()
        labelId = batch["label"].cuda()
        postId = batch["id"]
        
        outputs = model.forward(data)

        _, prediction = torch.max(outputs.data, 1)
 
        test_acc_top1 += torch.sum(prediction == labelId.data)
        
        for j in range(data.shape[0]):
            cm[labelId[j].item()][prediction[j].item()] += 1 
        
        if num_labels > 2:
            _, prediction3 = outputs.data.topk(3, 1, True, True)
            prediction3 = prediction3.t()
            test_acc_top3 += prediction3.eq(labelId.view(1,-1).expand_as(prediction3)).sum()
        
        if num_labels > 5:
            _, prediction5 = outputs.data.topk(5, 1, True, True)
            prediction5 = prediction5.t()
            test_acc_top5 += prediction5.eq(labelId.view(1,-1).expand_as(prediction5)).sum()
            
        total += data.shape[0]

    test_acc_top1 = test_acc_top1 / total
    test_acc_top3 = test_acc_top3 / total
    test_acc_top5 = test_acc_top5 / total
    
    print("Total Test Data Processed ==>{}".format(total))
    print(test_acc_top1)
    print(test_acc_top3)
    print(test_acc_top5)
    total_samples = np.sum(cm)
    correct = np.trace(cm)
    acc = correct/total_samples
    assert int(test_acc_top1.item()) == int(acc)