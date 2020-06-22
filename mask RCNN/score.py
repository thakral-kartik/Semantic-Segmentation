# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:44:52 2020

@author: Kartik
"""

from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    #print(y_true.dtype,y_pred.dtype)
    #assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def scores(actual, preds):
    p, l = [], []
    for labels,pred in zip(actual,preds):
        if pred=='None':
            pred=np.zeros((256,256))
    
        labels = labels
        pred = pred.flatten()
        labels = labels.flatten()
        p.extend(pred)
        l.extend(labels)
    print(np.array(l).shape, np.array(p).shape)
    score = f1_score(l, p, average='macro')
    print("f1 score / dice score = ", score)
    report = classification_report(l, p, target_names=['class 0', 'class 1'])
    print(report)
    print("pixAcc score = ",accuracy_score(l,p))
    print("confusion matrix : \n",confusion_matrix(l,p))
    
    ious=[]

    for  act, pred in zip(actual, preds):
        if pred == 'None':
            pred = np.zeros((256,256))
            ious.append(IoU(act.astype(float), pred.astype(float)))
        else:
            #pred=np.zeros((256,256))
            ious.append(IoU(act.astype(float), pred.astype(float)))
    print("Total predictions:", len(preds))
    ious = np.array(ious)
    print("Mean IOU :")
    print(round(ious.mean(), 4))