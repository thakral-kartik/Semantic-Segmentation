import os
import glob
from scipy import io
import numpy as np
import skimage.io
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
#baseline+transposed_conv+max_pooling_(1).ipynb

import torch
import torchvision
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import dataloader

def create_path(path):
    try:
        path = path.split('\\')[-1]
        print("path:", path)
        if not os.path.isdir('predictions'):
            os.mkdir('predictions')
        if not os.path.isdir(os.path.join('predictions', path)):    
            os.mkdir(os.path.join('predictions', path))
        
        return True
    except:
        return False

def write_pred(model, testloader, path, device, name_list):
    count, ext = 0, '.png'

    if not create_path(path):
        print("Error creating prediction directory")
        return False
    
    for batch in testloader:
        images, labels = batch
        #images = batch[0]
        
        images = images.to(device)
        pred = model(images)
        #print(path)
        for xray, img, label in zip(images, pred, labels):
        #for xray, img in zip(images, pred):
            img = torch.round(img) #thresholding
            #print(xray.shape)
            xray = xray.permute(1, 2, 0)
            xray = xray.squeeze(0).cpu().detach().numpy()
            img = img.squeeze(0).cpu().detach().numpy()
            label = label.numpy()
            
            xray = xray * 255.0
            img = img * 255.0
            label = label * 255.0
            
            #cv2.imwrite(os.path.join(path, name_list[count]+'_xray'+ext), xray)
            cv2.imwrite(os.path.join(path, name_list[count]), img)
            #cv2.imwrite(os.path.join(path, name_list[count]+'_label'+ext), label)
            
            count += 1
    return True

def load_model(model_name):
    if model_name == 'unet':
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)

    elif model_name == 'mask_rcnn':
        #model = mymodel()
        num_classes = 2
        # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
        
    model.load_state_dict(torch.load(os.path.join('models', 'model_pretrained_state_' + model_name + '_bce.pt')))
    return model

def main():
    model_name = 'unet'
    #model_name = 'mask_rcnn'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_path = os.path.join('predictions', 'exp_10_covid_' + model_name)
    print("model:", model_name, "\tDevice:", device)
    
    x_val, y_val, name_list = dataloader.get_new_dataset()
    print("Data loaded..")
    
    x_val, y_val = torch.Tensor(x_val), torch.Tensor(y_val)
    x_val = x_val.permute(0, 3, 1, 2)
    
    #trainset = data.TensorDataset(x_train, y_train)
    #valset = data.TensorDataset(x_val, y_val)
    valset = data.TensorDataset(x_val)
    
    #trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False)
    testloader = torch.utils.data.DataLoader(valset, batch_size=10, shuffle=False)
    
    model = load_model(model_name)
    if torch.cuda.is_available():
        model = model.to(device)
        #testloader = testloader.to(device)
    print("Model successfully loaded")

    model.eval()

    if not write_pred(model, testloader, pred_path, device, name_list):
        print("error saving the prediction on testset..")
    else:
        print("Test predictions successfully saved..")

    print("Done")
    return model, testloader

if __name__ == '__main__':
	model, testloader = main()