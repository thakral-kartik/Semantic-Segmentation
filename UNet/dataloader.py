# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:21:08 2020

@author: Kartik
"""

import numpy as np
import glob
import os
import skimage.io
import scipy.io

from sklearn.utils import shuffle

def modify_mask(mask):
    for i in range(mask.shape[0]):
        for j in range(mask.shape[0]):
            if mask[i][j] >= 1: mask[i][j] = 1.0
            else: mask[i][j] = 0.0
    return mask

def get_split(path):
    x, y = [], []
    image_name, name_mask = [], []
    #for folder in glob.glob(os.path.join(path, '*')):
    for image, mask in zip(glob.glob(os.path.join(path, 'xray', '*')), glob.glob(os.path.join(path, 'mask', '*'))):
        img = skimage.io.imread(image)
        d = scipy.io.loadmat(mask)
        mask_name = (image.split('\\')[-1]).split('.')[0] + '_mask'
        if '_mirror_' in mask_name:
            l = mask_name.split('_')
            mask_name = l[0] + '_' + l[1] + '_' + l[3]
        #print(d)
        x.append(img)
        y.append(modify_mask(d[mask_name]))
        #print(mask_name)
        #print(d[mask_name])
        #exit(0)
        image_name.append(image.split('\\')[-1])
        name_mask.append(mask.split('\\')[-1])

    return np.array(x), np.array(y), image_name, name_mask
            
def create_healthy_masks(count, shape):
    masks = []
    for i in range(count):
        masks.append(np.zeros(shape))
        
    return masks

def get_new_dataset():
    '''
    x_train, y_train, x_val, y_val, img_list_1, mask_list_1, img_list_2, mask_list_2 = [], [], [], [], [], [], [], []
    for split in glob.glob(os.path.join('modified_dataset_256', '*')):
        if 'train' in split:
            print("split:", split)
            x_train, y_train, img_list_1, mask_list_1 = get_split(split)
            print(len(img_list_1), len(mask_list_1))
        elif 'val' in split:
            print("split:", split)
            x_val, y_val, img_list_2, mask_list_2 = get_split(split)
            print(len(img_list_2), len(mask_list_2))
        else:
            print("Unknown split:", split)
            exit(0)
    #img_list_1.extend(img_list_2)
    #mask_list_1.extend(mask_list_2)
    
    try:
        f1 = open("image_list_train.txt", "w")
        f2 = open("mask_list_train.txt", "w")
        f3 = open("image_list_val.txt", "w")
        f4 = open("mask_list_val.txt", "w")
        for i, j in zip(img_list_1, mask_list_1):
            f1.write(i+",")
            f2.write(j+",")
        for k, l in zip(img_list_2, mask_list_2):
            f3.write(k+",")
            f4.write(l+",")
        print("Files successfully written")
    except:
        print("Error writing files")
    
    #return x_train, y_train, x_val, y_val, img_list_1, mask_list_1, img_list_2, mask_list_2
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_val.npy', x_val)
    np.save('y_val.npy', y_val)
    '''
    
    x_train = np.load(os.path.join('npys', 'x_train.npy'))
    #train_h = np.load(os.path.join('npys', 'train_healthy.npy'))
    
    x_test = np.load(os.path.join('npys', 'x_test.npy'))
    #test_h = np.load(os.path.join('npys', 'test_healthy.npy'))
    
    y_train = np.load(os.path.join('npys', 'y_train_mask_.npy'))
    #train_m = np.array(create_healthy_masks(x_train.shape[0], (256, 256)))
    
    y_test = np.load(os.path.join('npys', 'y_test_mask.npy'))
    #test_m = np.array(create_healthy_masks(x_test.shape[0], (256, 256)))
    
    name_list = np.load(os.path.join('npys', 'name_list.npy'))
    
    #x_train = np.concatenate((x_train, train_h), axis=0)
    #y_train = np.concatenate((y_train, train_m), axis=0)
    #x_test = np.concatenate((x_test, test_h), axis=0)
    #y_test = np.concatenate((y_test, test_m), axis=0)
    
    #x_train, y_train = shuffle(x_train, y_train)
    #x_test, y_test = shuffle(x_test, y_test)
	
    return x_train, y_train, x_test, y_test, name_list