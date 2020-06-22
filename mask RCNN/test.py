# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:05:10 2020

@author: Kartik
"""

import cv2
import os
from skimage import io
import numpy as np
import torch
import torchvision
from PIL import Image
import utils
import transforms as T
from engine import train_one_epoch, evaluate

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import model
import config
import dataloader

def check_dir():
    if not os.path.sidir('predictions'):
        os.mkdir('predictions')
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)

def write_preds(actual, preds, x_val):
    actual=np.array(actual)
    preds=np.array(preds)
    
    #os.mkdir('./Test_set_disease/preds_proper_')
    pred_path = os.path.join('predictions', 'exp_'+str(config.exp_num))
    check_dir(pred_path)
    d, count = 0, 0
    
    for act, pred, mask in zip(actual, preds, x_val):
        if pred=='None':
            pred=np.zeros((256,256))
        d+=1
        #s1='actual_'+str(d)+'.jpg'
        #s2='pred_'+str(d)+'.jpg'
        s1=image_names[count].split(".")[0]+str('_act')+'.jpg'
        s2=image_names[count].split(".")[0]+str('_pred')+'.jpg'
        s3=image_names[count].split(".")[0]+str('_label')+'.jpg'
        #plt.imshow(act)
        io.imsave(os.path.join(pred_path, s1), mask)
        io.imsave(os.path.join(pred_path, s2), pred)
        #cv2.imwrite("./Test_set_disease/preds_proper_/"+s3,act)
        io.imsave(os.path.join(pred_path, s3), act*255)
        count+=1
    
    print(count)

def evaluate(dataset_test, model):
    preds, actual, image_name, all_preds, count = [], [], [], [], 0
    model.eval()
    for i in range(len(dataset_test)):
        test_image, _ = dataset_test[i]
        actual.append(_['actual_mask'])
        image_names.append(_["image_name"])
        #test_image = torch.Tensor(test)
        #test_image = test_image.permute(2,0,1)
        prediction = model([test_image.to(device)])
        #all_preds.append(prediction)
        print(prediction[0]['masks'].size())
        
        if prediction[0]['masks'].size()[0] == 0:
           preds.append(np.zeros((256,256)))
           #preds.append('None')
           count+=1
           print("count: ",count)
           continue
        #actual.append(_['actual_mask'])
        new_image = torch.round(prediction[0]['masks'][0,0])
        new_image = new_image.cpu().detach().numpy()
        preds.append(torch.round(prediction[0]['masks'][0, 0]).cpu().detach().numpy())
    
    #all_preds=np.array(all_preds)
    actual = np.array(actual)
    preds = np.array(preds)
    print("total preds :", preds.shape)
    print("not detected any bounding box in images: ", count)
    
    model.eval()
    
    write_preds(actual, preds, x_val)
    #score.scores(actual, preds)

def load_model(model, count):
    #model.load_state_dict(torch.load(os.path.join('models', 'model_pretrained_disease_segmentation_state'+str(count)+'.pt')))
    #or load any specific model from /models directory
    model.load_state_dict(torch.load(os.path.join('models', 'model_pretrained_state_mask_rcnn.pt')))
    return model

def main(device, count):
    #instanciate the model
    model = load_mask_rcnn(config.num_classes)
    
    model = get_model_instance_segmentation(num_classes)
    
    model.to(device)
    
    model = load_model(model, count)
    
    x_val, y_val, name_list = dataloader.get_test_dataset
    
    dataset_test = dataloader.CXRTestDataset(x_val, y_val, name_list, dataloader.get_transform(train=False))
    #data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, collate_fn=utils.collate_fn)
    
    evaluate(test_loader, model)
    
if __name__ == '__main__':
    main(config.device, config.exp_num)