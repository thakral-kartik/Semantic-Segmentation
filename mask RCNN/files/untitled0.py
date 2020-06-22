# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:31:07 2020

@author: Lenovo
"""

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import utils
import transforms as T
from engine import train_one_epoch, evaluate
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

class CXRDataset(object):
    
    def __init__(self,new_x_train,new_y_train):
        
        self.imgs=new_x_train
        self.masks=new_y_train
    
    def __getitem__(self,idx):
        
        img=new_x_train[idx]
        mask=new_y_train[idx]
        
        obj_ids=np.unique(mask)
        obj_ids=obj_ids[1:]
        
        #split the color-encoded masks into a set
        masks=mask==obj_ids[:,None,None]
        
        num_objs=len(obj_ids)
        boxes=[]
        
        for i in range(num_objs):
            pos=np.where(masks[i])
            xmin=np.min(pos[1])
            xmax=np.max(pos[1])
            ymin=np.min(pos[0])
            ymax=np.max(pos[0])
            boxes.append([xmin,ymin,xmax,ymax])
        
        boxes=torch.as_tensor(boxes,dtype=torch.float32)
        labels=torch.ones((num_objs,),dtype=torch.int64)
        masks=torch.as_tensor(masks,dtype=torch.uint8)
        
        image_id=torch.tensor([idx])
        
        area=(boxes[:,3]-boxes[:,1])*(boxes[:,2]-boxes[:,0])
        
        iscrowd=torch.zeros((num_objs,),dtype=torch.int64)
        
        target={}
        target["boxes"]=boxes
        target["labels"]=labels
        target["masks"]=masks
        target["image_id"]=image_id
        target["area"]=area
        target["iscrowd"]=iscrowd
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img,target

    def __len__(self):
        return len(self.imgs)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
    return model



class PennFudanDataset(object):
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
        masks = mask == obj_ids[:, None, None]

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

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        print("target : ",target)
        return img, target

    def __len__(self):
        return len(self.imgs)

x_train=np.load('x_train.npy')
y_train=np.load('y_train.npy')

x_val=np.load('x_val.npy')
y_val=np.load('y_val.npy')

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
    


dataset=PennFudanDataset(new_x_train,new_y_train,get_transform(train=True))

#data_loader=torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=utils.collate_fn)
'''
num_classes=2
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x) 
'''
model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features=model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor=FastRCNNPredictor(in_features,num_classes)
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),aspect_ratios=((0.5, 1.0, 2.0),))


roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],output_size=7,sampling_ratio=2)
model = FasterRCNN(backbone,num_classes=2,rpn_anchor_generator=anchor_generator,box_roi_pool=roi_pooler)                                   

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 2

model = get_model_instance_segmentation(num_classes)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

num_epochs = 10

for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        #evaluate(model, data_loader_test, device=device)