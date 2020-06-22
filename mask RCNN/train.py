# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:30:01 2020

@author: Kartik
"""

import os
import numpy as np
import torch
import torchvision
from PIL import Image
import utils
import transforms as T
from engine import train_one_epoch, evaluate

import model
import config
import dataloader

'''
#install these
pip install cython
# Install pycocotools, the version by default in Colab
# has a bug fixed in https://github.com/cocodataset/cocoapi/pull/354
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
'''

def save_model(count):
    try:
        torch.save(model.state_dict(), os.path.join('models', 'model_pretrained_disease_segmentation_state'+str(count)+'.pt'))
        return True
    except:
        return False

def main(device, count):
    x_train, y_train = dataloader.get_train_dataset
    
    dataset = dataloader.CXRTrainDataset(x_train, y_train, dataloader.get_transform(train=True))
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=utils.collate_fn)
    
    model = models.load_mask_rcnn(config.num_classes)
    model = models.get_model_instance_segmentation(num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002,momentum=0.9)
        # and a learning rate scheduler
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
    #os.mkdir('lung_segmentation_preds')
    
    for epoch in range(config.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            #lr_scheduler.step()
            torch.save(model.state_dict(),'./lung_segmentation_preds/model_pretrained_CXR.pt')
            # evaluate on the test dataset
            #evaluate(model, data_loader_test, device=device)
    
    
    print("\nTraining complete.")
    
    if not save_model(count):
        print("Error saving model")
    else:
        print("Model successfully saved")
    

if __name__ == '__main__':
    main(config.device, config.exp_num)