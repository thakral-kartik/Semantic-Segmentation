#Which experiment is it?
exp_num = 1

#Number of classes in semantic segmentation
num_classes = 2

#cpu or gpu
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#number of qpochs for training
num_epochs = 50