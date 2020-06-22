# -*- coding: utf-8 -*-
"""
Created on Sun May 10 20:13:59 2020

@author: Kartik
"""

import os
import numpy as np
import transforms as T
from PIL import Image


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def check_unique(x, y):
    new_x_train=[]
    new_y_train=[]
    
    index=0
    for mask in y_train:
        
        if len(np.unique(mask))==1:
            index+=1
            continue
        new_x_train.append(x_train[index])
        new_y_train.append(y_train[index])
        index+=1
    return new_x_train, new_y_train

def get_train_dataset():
    x_train = np.load(os.path.join('npys', 'x_train.npy'))
    y_train = np.load(os.path.join('npys', 'y_train.npy'))
    
    x_train, y_train = check_unique(x_train, y_train)
        
    return x_train, y_train

def get_test_dataset():
    x_test = np.load(os.path.join('npys', 'x_test_covid.npy'))
    y_test = np.load(os.path.join('npys', 'y_test.npy'))
    name_list = np.load(os.path.join('npys', 'x_test_name_list_covid.npy'))
    
    x_test, y_test = check_unique(x_test, y_test)
    
    return x_test, y_test, name_list

class CXRTrainDataset(object):
    def __init__(self, new_x_train,new_y_train,transforms):
       
        self.imgs = new_x_train
        self.masks = new_y_train
        self.transforms=transforms
        
    def __getitem__(self, idx):
        # load images ad masks
        #img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        #mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        #img = Image.open(img_path).convert("RGB")
        img=self.imgs[idx]
        mask=self.masks[idx]
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        #mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = np.expand_dims(mask,axis=0)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #target["actual_mask"]=mask

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        #print("target : ",target)
        return img, target

    def __len__(self):
        return len(self.imgs)


class CXRTestDataset(object):
    def __init__(self, new_x_train,new_y_train,name_list,transforms):
       
        self.imgs = new_x_train
        self.masks = new_y_train
        self.names=name_list
        self.transforms=transforms
        
    def __getitem__(self, idx):
        # load images ad masks
        #img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        #mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        #img = Image.open(img_path).convert("RGB")
        img=self.imgs[idx]
        mask=self.masks[idx]
        name=self.names[idx]
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        #mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = np.expand_dims(mask,axis=0)

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["actual_mask"]=mask
        target["image_name"]=name

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        #print("target : ",target)
        return img, target

    def __len__(self):
        return len(self.imgs)