# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 13:15:50 2020

@author: Kartik
"""

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
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

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

def Classification_Report(label, pred):
    report = classification_report(label, pred, target_names=['class 0', 'class 1'])
    print(report)
    return report

def write_pred(model, testloader, path, device):
    count, ext = 0, '.png'
    
    if not create_path(path):
        print("Error creating prediction directory")
        return False
    
    for batch in testloader:
        images, labels = batch
        
        images = images.to(device)
        pred = model(images)
        #print(path)
        for xray, img, label in zip(images, pred, labels):
            img = torch.round(img) #thresholding
            #print(xray.shape)
            xray = xray.permute(1, 2, 0)
            xray = xray.squeeze(0).cpu().detach().numpy()
            img = img.squeeze(0).cpu().detach().numpy()
            label = label.numpy()
            
            img = img * 255.0
            label = label * 255.0
            
            cv2.imwrite(os.path.join(path, str(count)+'_xray'+ext), xray)
            cv2.imwrite(os.path.join(path, str(count)+'_pred'+ext), img)
            cv2.imwrite(os.path.join(path, str(count)+'_label'+ext), label)
            
            count += 1
    return True

def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0

    for inputs, labels in testloader:
        inputs = inputs.to(device)
        output = model(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

def IOU(model, testloader, loss, device):
    iou = []
    p, l = [], []
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        
        pred = pred.view(-1)
        labels = labels.view(-1)
        pred = pred.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        
        new_pred = [1 if i>=0.5 else 0 for i in pred]
        p.extend(new_pred)
        l.extend(labels)
        intersection = np.logical_and(l, p).sum()
        union = np.logical_or(l, p).sum()
        res = (intersection + 1) * 1. / (union + 1)
        iou.append(res)
            
    result = round(np.array(iou).mean(),4)
    print("Mean IOU: ", result)
    return result

def dice_score(model, testloader, loss, device):
    p, l = [], []
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        
        pred = pred.view(-1)
        labels = labels.view(-1)
        pred = pred.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        
        new_pred = [1 if i>=0.5 else 0 for i in pred]
        p.extend(new_pred)
        l.extend(labels)
    print(np.array(l).shape, np.array(p).shape)
    score = f1_score(l, p, average='macro')
    #iou = IOU(l, p)
    print("f1 score / dice score = ", score)
    return l, p, score

def pixel_vise_acc(model, testloader, loss):
    total_loss, count, acc, accuracy = 0, 0, 0, []
    batch_size = 8
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)
        pred = model(images)
        
        pred = pred.view(-1)
        labels = labels.view(-1)
        loss_ = loss(pred,labels)
        total_loss += loss_.item()
        
        new_pred = [1 if i>=0.5 else 0 for i in pred]
        
        count += 1
        labels = labels.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        for i, j in zip(new_pred, labels):
            if i==j:
                acc += 1
        accuracy.append(acc)
    print("sum: ", np.sum(np.array(accuracy)))
    return np.sum(np.array(accuracy))/(len(accuracy)*batch_size*256*256)
        
class InvSoftDiceLoss(nn.Module):

    '''
    Inverted Soft Dice Loss
    '''   
    def __init__(self, weight=None, size_average=True):
        super(InvSoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        logits = F.sigmoid(logits)
        iflat = 1-logits.view(-1)
        tflat = 1-targets.view(-1)
        intersection = (iflat * tflat).sum()
    
    
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class diceloss(torch.nn.Module):
    #https://discuss.pytorch.org/t/implementation-of-dice-loss/53552
    def init(self):
        super(diceLoss, self).init()
    def forward(self,pred, target):
       smooth = 1.
       iflat = pred.contiguous().view(-1)
       tflat = target.contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

def write_log(score, report, iou):
    try:
        f = open("log_bce.txt", "w")
        f.write("Evaluation using BCE loss\n\n")
        f.write("Dice score: " + str(score))
        f.write("\n\nFinal report:\n" + str(report))
        f.write("\n\nIOU:" + str(iou))
    except:
        print("Error writing log")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_path = os.path.join('predictions', 'exp_11_bce')
    
    x_train, y_train, x_val, y_val, name_list = dataloader.get_new_dataset()
    #print(np.unique(y_train))
    x_train = x_train/255.0
    x_val = x_val/255.0
    print("Data loaded..")
    x_train, y_train, x_val, y_val = torch.Tensor(x_train), torch.Tensor(y_train), torch.Tensor(x_val), torch.Tensor(y_val)
    x_train = x_train.permute(0, 3, 1, 2)
    x_val = x_val.permute(0, 3, 1, 2)
    
    trainset = data.TensorDataset(x_train, y_train)
    valset = data.TensorDataset(x_val, y_val)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
    testloader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=False)
    
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=1, init_features=32, pretrained=True)
    #model.load_state_dict(torch.load('model_pretrained_state.pt'))
    
    if torch.cuda.is_available():
        trainloader = trainloader
        testloader = testloader
        model = model.to(device)
    
    if not os.path.isdir('models'):
            os.mkdir('models')
    
    model.train()
    m = nn.Sigmoid()
    loss = nn.BCELoss().cuda() #for binary classification
    #loss1 = diceloss().cuda()
    #loss2 = InvSoftDiceLoss().cuda() 
    #loss = nn.BCEWithLogitsLoss().cuda()
    #loss = diceloss().cuda()
    
    loss_list = []
    optimizer = optim.SGD(model.parameters(), lr=0.03)
    model_count = 0
    for epochs in range(100):
        total_loss=0
        count = 0
        for batch in trainloader:
            count += 1
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            #pred_list.append(preds)
            #print(preds.shape)
            #print(preds.shape,labels.shape)
            preds = preds.view(-1)
            #preds = m(preds)
            labels = labels.view(-1)
            #labels = labels.float()
            #loss_1 = loss1(preds, labels)
            #loss_2 = loss2(preds, labels)
            loss_ = loss(preds,labels)
            #print(loss_)
            #loss = loss1(preds,labels)+loss2(preds,labels)+loss3(preds,labels)
            
            #loss_ = F.cross_entropy(preds, labels)
            #loss_ = loss(preds, labels)
            loss_list.append(loss_.item())
            
            optimizer.zero_grad()
            if count%50==0:
                print("count : ", loss_.item())
            loss_.backward()     #calculate the gradients
            optimizer.step()    #weights update
            total_loss += loss_.item()
            
        print("Epoch :",epochs,"\tTotal loss:", total_loss/count)
        if model_count % 20 == 0:
            torch.save(model.state_dict(), os.path.join('models','model_pretrained_state_'+str(model_count)+'.pt'))
        model_count += 1
    
    torch.save(model.state_dict(), 'model_pretrained_state_bce.pt')
    print("Model state has been saved.")
    
    model.eval()
    if not write_pred(model, testloader, pred_path, device):
        print("error saving the prediction on testset..")
    else:
        print("Test predictions successfully saved..")
    
    #print("model accuracy: ", pixel_vise_acc(model,testloader, loss))
    #print("Model.evluate: ", validation(model, testloader, loss))
    #l, p = classification_report(masks,pred, target_names=['class 0','class 1'])
    l, p, score = dice_score(model, testloader, loss, device)
    report = Classification_Report(l, p)
    iou = IOU(model, testloader, loss, device)
    
    write_log(score, report, iou)
    
    print("Done")
    return model, testloader, loss, device
    
if __name__ == '__main__':
    model, testloader, loss, device = main()
    