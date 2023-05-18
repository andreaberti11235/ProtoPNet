#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 09:15:42 2022

versione fine tuning


@author: si-lab
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, models, transforms
from torch.nn.parallel import DistributedDataParallel as ddp 
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
from tqdm import tqdm
from time import gmtime,strftime


data_path = os.path.join(os.getcwd(),'datasets') #
train_dir = os.path.join(data_path,'push_augmentor') #
#test_dir = os.path.join(data_path,'test_augmented') #'valid/' #
test_dir = os.path.join(data_path,'valid_augmented') #'valid/' #

#TODO prenderli corretamente col rispettivo valore calcolato:
# mean = np.float32(np.uint8(np.load(os.path.join(data_path,'mean.npy')))/255)
# std = np.float32(np.uint8(np.load(os.path.join(data_path,'std.npy')))/255)
mean = 0.5
std = 0.5

from sklearn.metrics import accuracy_score
# import seaborn as sn
# import pandas as pd

img_size = 224 #564 #224 #TODO
num_epochs = 1000 #TODO




parse = argparse.ArgumentParser(description="")

parse.add_argument('model_name', help='Name of the baseline architecture: alexnet, resnet18, resnet34, resnet50, vgg..',type=str)
parse.add_argument('num_layers_to_train', help='Number of contigual conv2d layers to train beginning from the end of the model up',type=int)
parse.add_argument('lr', help='Learning rate',type=float)
parse.add_argument('wd', help='Weight decay',type=float)
parse.add_argument('dr', help='Dropout rate to be added after fully connected layers',type=float)
parse.add_argument('d2Dr', help='Dropout 2D rate to be added after Conv2D layers',type=float)
# parse.add_argument('num_dropouts',help='Number of dropout layers in the bottleneck of ResNet18, if 1 uses one, if 2 uses two.', type=int)
#Add string of information about the specific experiment run, as dataset used, images specification, etc
parse.add_argument('run_info', help='Plain-text string of information about the specific experiment run, as the dataset used, the images specification, etc. This is saved in run_info.txt',type=str)
parse.add_argument('-nd2d', '--num_dropout_2d', default=1, type=int, help='For ResnNet. Number of Convolutional layers after which to add a Dropout layer (default is 1).')
parse.add_argument('-o', '--optimiser', default='adam', type=str, help='Specify the name of the optimiser (default is adam). Possible options: adam, sgd, rms_prop.')
parse.add_argument('-p', '--pretrained', help='Add this flag to use the pre-trained model.', action='store_true')

args = parse.parse_args()

num_layers_to_train = args.num_layers_to_train
actual_model_name = args.model_name
model_names = [actual_model_name+f'_finetuning_last_{num_layers_to_train}_layers_{img_size}_imgsize']#TODO clahe?

joint_lr_step_size = 10
gamma_value = 0.5

lr = [args.lr]
wd = [args.wd]
dropout_rate = [args.dr]
dropout2d_rate = [args.d2Dr]
# num_dropouts = args.num_dropouts
run_info_to_be_written = args.run_info

num_d2d = args.num_dropout_2d
optimiser = args.optimiser

pretrained = args.pretrained
#pretrained = True

# lr=[1e-6]
# wd = [1e-3] #[5e-3]
# dropout_rate = [0.5]

batch_size = [10] #TODO 
batch_size_valid = 2
# joint_lr_step_size = [2, 5, 10]
# gamma_value = [0.10, 0.50, 0.25]

num_classes = 1
# window = 20
# patience = int(np.ceil(50/window)) # sono 3*20 epoche ad esempio
window = 5
patience = int(np.ceil(12/window)) #3 


print('CUDA visible devices, before and after setting possible multiple GPUs (sanity check):')
# print(os.environ['CUDA_VISIBLE_DEVICES'])
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' #TODO
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(os.environ['CUDA_VISIBLE_DEVICES'])

os_env_cudas = os.environ['CUDA_VISIBLE_DEVICES']
os_env_cudas_splits = os_env_cudas.split(sep=',')
workers = 4*len(os_env_cudas_splits) #TODO METTERE 4* QUANDO POSSIBILE


import random
torch.cuda.empty_cache()
# SEED FIXED TO FOSTER REPRODUCIBILITY
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(seed=1)




N = len(lr) * len(wd) * len(dropout_rate) * len(batch_size) * len(dropout2d_rate)
#* len(joint_lr_step_size) * len(gamma_value)

# def get_N_HyperparamsConfigs(N=0, lr=lr, wd=wd, joint_lr_step_size=joint_lr_step_size,
#                              gamma_value=gamma_value):
def get_N_HyperparamsConfigs(N=0, lr=lr, wd=wd, dropout_rate=dropout_rate, dropout2d_rate=dropout2d_rate):
    configurations = {}
    h = 1
    for i in lr:
        for j in wd:
            for k in dropout_rate:
                for l in batch_size:
                    for m in dropout2d_rate:
                        configurations[f'config_{h}'] = [i, j, k, l, m]
                        h += 1
    
                     
    configurations_key = list(configurations.keys())
    chosen_configs = sorted(random.sample(configurations_key,N)) 
    return [configurations[x] for x in chosen_configs]


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False, window=20, patience=3):
    since = time.time()
    val_acc_history = []
    train_acc_history = []
    val_loss=[]
    train_loss=[]
    count = 0
    prima_volta = True
    to_be_stopped = False
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stop_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            y_true = []
            y_pred = []

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                
                labels=labels.float()
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, labels.unsqueeze(1))                       

                    # preds = np.round(outputs.detach().cpu())
                    # preds = torch.round(outputs)
                    
                    # PREDICTIONS 
                    pred = np.round(outputs.detach().cpu())
                    target = np.round(labels.detach().cpu())             
                    y_pred.extend(pred.tolist())
                    y_true.extend(target.tolist())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == torch.round(labels.data))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            epoch_acc = accuracy_score(y_true,y_pred)

            with open(os.path.join(output_dir,f'{phase}_metrics.txt'),'a') as f_out:
                      f_out.write(f'{epoch},{epoch_loss},{epoch_acc}\n')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                torch.save(obj=model, f=os.path.join(output_dir,('epoch_{}_acc_{:.4f}.pth').format(epoch, best_acc)))
                
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss.append(epoch_loss)
                
                ## EARLY STOPPING
                if ((epoch+1) >= 2*(window)) and ((epoch+1) % window==0):
                    
                    early_stop_acc = epoch_acc
                    
                    loss_npy = np.array(val_loss,dtype=float)
                    windowed = [np.mean(loss_npy[i:i+window]) for i in range(0,len(val_loss),window)]
                    # windowed_epochs = range(window,len(val_loss)+window,window)
                    windowed_2 = windowed[1:]
                    windowed_2.append(0.0)
                    deriv_w = [(e[0]-e[1])/window for e in zip(windowed,windowed_2)]
                    deriv_w = deriv_w[:-1]
                    
                    if prima_volta:
                        prima_volta = False
                        thresh = deriv_w[0]*0.10 
                    
                    if deriv_w[-1] < thresh:
                        print(f'DETECTION DI VALORE SOTTOSOGLIA, con soglia {thresh}')
                        count += 1
                        if count ==1:
                            model_wts_earlyStopped = copy.deepcopy(model.state_dict())                        
                            torch.save(obj=model, f=os.path.join(output_dir,('earlyStopped_epoch_{}_acc_{:.4f}.pth').format(epoch, epoch_acc)))
                            print('Modello salvato per quando verrà stoppato allenamento')
                        if count >= patience:
                            to_be_stopped = True
                            print(f'PAZIENZA SUPERATA EPOCA {epoch}, ESCO')
                            break
                        else:
                            print(f'EPOCA {epoch} - ASPETTO ANCORA IN PAZIENZA ({patience-count}) EPOCHE')
                            continue
                        
                        
                    else:
                        count = 0
                
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss.append(epoch_loss)
                joint_lr_scheduler.step()
        if to_be_stopped:
            break
        
        print()
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    if to_be_stopped==False: #the case when EarlyStop never occurs
        model.load_state_dict(best_model_wts)
    else:
        model.load_state_dict(model_wts_earlyStopped)
        best_acc = early_stop_acc

    return model, val_acc_history, train_acc_history, val_loss, train_loss, best_acc





def set_parameter_requires_grad(model, feature_extracting, num_layers_to_train, dropout2d_rate):
    # # Versione: decongelare un numero desiderato di layer a partire dal fondo
    # if feature_extracting:
    #     t = 0
    #     for child in model.modules():
    #         if isinstance(child,nn.Conv2d):
    #             t+=1
    #     print(f'Total number of Conv2d layers in the model: {t}')
        
    #     c = 0
    #     for child in model.modules():
    #         for param in child.parameters(): #first, freeze all the layers from the top
    #             param.requires_grad = False
                
    #         if isinstance(child,nn.Conv2d):
    #             c+=1
    #         if c > t - num_layers_to_train: #un-freeze all the following layers (conv2d & bn)
    #             for param in child.parameters():
    #                 param.requires_grad = True
    
    
    
    ## Versione per introdurre Dropout2d intermedi frai vari Conv2d
    if feature_extracting:
        t = 0
        for child in model.modules():
            if isinstance(child,nn.Conv2d):
                t+=1
                
        print(f'Conv2d layers re-trained in the model: {num_layers_to_train}/{t}')
        
        c = 0
        c_dropout = 0
        is_first_time = True
        
        #
        for name,child in model.named_modules():
            splits = name.split('.')
            
            if is_first_time:
                is_first_time = False
                ## TODO: Versione tutta congelata e alcuni decongelati
                for param in child.parameters(): #first, freeze all the layers from the top
                   
                    param.requires_grad = False
   
                    
                # ## TODO: Versione tutta viene riallenata
                # for param in child.parameters():
                #     if param.requires_grad is not True:
                #         print(f'{name} non era TRUE')
                #         param.requires_grad = True
                
            if isinstance(child,nn.Conv2d):
                c+=1
            
            assert(t>=num_layers_to_train)
            
            if c > t - num_layers_to_train: #un-freeze all the following layers (conv2d & bn)
                for param in child.parameters():
                    param.requires_grad = True         
        
        
        model_copy = copy.deepcopy(model)
        
        for name,child in model_copy.named_modules():
            splits = name.split('.')

                
            if isinstance(child,nn.Conv2d):
                c+=1
            
            assert(t>=num_layers_to_train)
            
            if c > t - num_layers_to_train: #un-freeze all the following layers (conv2d & bn)
                if model._get_name() == 'ResNet':
                    if isinstance(child,nn.Conv2d) and splits[-1]=='conv1': #TODO conv1
                    
                        c_dropout += 1
                        
                        # if c_dropout <= 3: #TODO i primi 3
                        #     new_module = nn.Sequential(
                        #             child,
                        #             nn.Dropout2d(p=dropout_rate))
        
                        #     if len(splits)==1:
                        #         setattr(model, name, new_module)
                        #     elif len(splits)==3:
                        #         setattr(getattr(model,splits[0])[int(splits[1])], splits[2], new_module)
                        #     # #
                        #     # elif len(splits)==4:
                        #     #     setattr(getattr(getattr(model,splits[0])[int(splits[1])],splits[2]), splits[3], new_module)
                        
                        if c_dropout > 9 - num_d2d: # hardcode 9 e 3; TODO gli ultimi 3 (9 è il numero totale di conv1)
                        #if c_dropout > 9 - 2: # hardcode 9 e 3; TODO gli ultimi 3 (9 è il numero totale di conv1)
                            # modificato 3 con 1 per aggiungere 1 solo layer di dropout; cambiato 1 con 2
                            new_module = nn.Sequential(
                                    child,
                                    nn.Dropout2d(p=dropout2d_rate))
        
                            if len(splits)==1:
                                setattr(model, name, new_module)
                            elif len(splits)==3:
                                setattr(getattr(model,splits[0])[int(splits[1])], splits[2], new_module)      
                elif model._get_name() == 'AlexNet':
                   if isinstance(child,nn.Conv2d): #TODO conv1
                   
                        c_dropout += 1
                        
                        if c_dropout > 5 - 3: # hardcode 5 e 1; TODO aggiunta dropout agli ultimi 1 layer (5 è il numero totale di conv2d)
                            new_module = nn.Sequential(
                                    child,
                                    nn.Dropout2d(p=dropout_rate))
                            setattr(model.features, splits[1], new_module)

        # ## Versione iniziale:
        # for param in model.parameters():
        #     param.requires_grad = False
        
           
            
def initialize_model(model_name, num_classes, feature_extract, dropout_rate, num_layers_to_train, dropout2d_rate, use_pretrained=True):
# def initialize_model(model_name, num_classes, feature_extract, dropout_rate, num_dropouts, num_layers_to_train, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract, num_layers_to_train, dropout2d_rate=dropout2d_rate)
        num_ftrs = model_ft.fc.in_features
        
        ##Version 1
        model_ft.fc = nn.Sequential( # replace the original fully connected
            nn.Dropout(p=dropout_rate), #TODO added some dropout
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
            )
        
        # ##Version 2
        # #TODO added bottleneck-layers and some dropout 
        # if num_dropouts==1:
        #     model_ft.fc = nn.Sequential(
                
        #         nn.Linear(num_ftrs,64), #512>64
        #         nn.ReLU(),
                
        #         nn.Dropout(p=dropout_rate),                 
        #         nn.Linear(64, num_classes),
        #         nn.Sigmoid()
        #         )
        # elif num_dropouts==2:
        #     model_ft.fc = nn.Sequential(
                
        #         nn.Dropout(p=dropout_rate),                
        #         nn.Linear(num_ftrs,64), #512>64
        #         nn.ReLU(),
                
        #         nn.Dropout(p=dropout_rate),                 
        #         nn.Linear(64, num_classes),                
        #         nn.Sigmoid()
        #         )
        
        
        
        # #TODO 11 aprile 2022: idea di semplificare la base architecture per ridurre il numero di out_features uscente e di conseguenza il numero di filtri necessari ai successivi layer FC
        # if num_dropouts==1:
        #     model_ft.fc = nn.Sequential(
    
        #         #Fully connected
        #         nn.Linear(num_ftrs,256),
        #         nn.ReLU(),
                
        #         nn.Linear(256,128),
        #         nn.ReLU(),
                
        #         #Dropout
        #         nn.Dropout(p=dropout_rate),
                
        #         # Classification layer
        #         nn.Linear(128, num_classes),
        #         # nn.Softmax() 
        #         nn.Sigmoid()
        #         )
        
        # elif num_dropouts==2:
        #     model_ft.fc = nn.Sequential(

        #         #Fully connected
        #         nn.Linear(num_ftrs,256),
        #         nn.ReLU(),
                
        #         nn.Dropout(p=dropout_rate),

        #         nn.Linear(256,128),
        #         nn.ReLU(),
                
        #         #Dropout
        #         nn.Dropout(p=dropout_rate),
                
        #         # Classification layer
        #         nn.Linear(128, num_classes),
        #         # nn.Softmax() 
        #         nn.Sigmoid()
        #         )
            
        # else:
        #     print('Attention please, invalid value for num_dropouts')
        #     raise ValueError
            
            
            
        #     # #Fully connected
        #     # nn.Linear(num_ftrs,128),
        #     # nn.ReLU(),
            
        #     # # nn.Dropout(p=dropout_rate), #TODO
            
        #     # nn.Linear(128,10),
        #     # nn.ReLU(),
            
        #     # #Dropout
        #     # nn.Dropout(p=dropout_rate),
            
        #     # # Classification layer
        #     # nn.Linear(10, num_classes),
        #     # # nn.Softmax() 
        #     # nn.Sigmoid()
        #     # )
        
        
            
        #     # #Fully connected
        #     # nn.Linear(num_ftrs,10),
        #     # nn.ReLU(),
            
        #     # #Dropout
        #     # nn.Dropout(p=dropout_rate),
            
        #     # # Classification layer
        #     # nn.Linear(10, num_classes),
        #     # # nn.Softmax() 
        #     # nn.Sigmoid()
        #     # )
        
        input_size = img_size  

    if model_name == "resnet34":
        """ Resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract, num_layers_to_train, 
                                    dropout2d_rate=dropout2d_rate)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes),
            nn.Sigmoid()
            )
        
        # #TODO 8 aprile 2022: idea di semplificare la base architecture per ridurre il numero di out_features uscente e di conseguenza il numero di filtri necessari ai successivi layer FC
        # model_ft.fc = nn.Sequential(

        #     #Fully connected
        #     nn.Linear(num_ftrs,256),
        #     nn.ReLU(),
                       
        #     nn.Linear(256,128),
        #     nn.ReLU(),
            
        #     #Dropout
        #     nn.Dropout(p=dropout_rate),
            
        #     # Classification layer
        #     nn.Linear(128, num_classes),
        #     # nn.Softmax() 
        #     nn.Sigmoid()
        #     )
            
            
            
        #     # # #TODO consiglio di fare 512>128>10>1. Fully connected
        #     # nn.Linear(num_ftrs,128),
        #     # nn.ReLU(),
                       
        #     # nn.Linear(128,10),
        #     # nn.ReLU(),
            
        #     # # #Dropout
        #     # nn.Dropout(p=dropout_rate),
            
        #     # # # Classification layer
        #     # nn.Linear(10, num_classes),
        #     # # # nn.Softmax() 
        #     # nn.Sigmoid()
        #     # )
            
            
        #     #TODO 14 aprile 2022 mattina
        #     #nn.Linear(num_ftrs,20),
        #     #nn.ReLU(),
            
        #     #Dropout
        #     #nn.Dropout(p=dropout_rate),
            
        #     # Classification layer
        #     #nn.Linear(20, num_classes),
        #     # nn.Softmax() 
        #     #nn.Sigmoid()
        #     #)
        
        input_size = img_size  

    # if model_name == "resnet50":
    #     """ Resnet50
    #     """
    #     # model_ft = models.resnet50(pretrained=use_pretrained)
    #     # set_parameter_requires_grad(model_ft, feature_extract)
    #     # num_ftrs = model_ft.fc.in_features
    #     # model_ft.fc = nn.Linear(num_ftrs, num_classes)
    #     # input_size = img_size   
        
    #     model_ft = models.resnet50(pretrained=use_pretrained)
    #     set_parameter_requires_grad(model_ft, feature_extract, num_layers_to_train)
    #     num_ftrs = model_ft.fc.in_features
   
        
    #     model_ft.fc = nn.Sequential(

    #         #Fully connected
    #         #nn.Linear(num_ftrs,1024),
    #         #nn.ReLU(),
                       
    #         #nn.Linear(1024,512),
    #         #nn.ReLU(),
            
    #         #Dropout
    #         #nn.Dropout(p=dropout_rate),
            
    #         # Classification layer
    #         #nn.Linear(512, num_classes),
    #         # nn.Softmax() 
    #         #nn.Sigmoid()
    #         #)
            
    #         ## VERSIONE CON SOLO DUE FC E NON TRE:
    #         # #Fully connected
    #         nn.Linear(num_ftrs,512),
    #         nn.ReLU(),
                                  
    #         # #Dropout
    #         nn.Dropout(p=dropout_rate),
            
    #         # # Classification layer
    #         nn.Linear(512, num_classes),
    #         # # nn.Softmax() 
    #         nn.Sigmoid()
    #         )
        
    #     input_size = img_size  
    
    if model_name == "vgg19":
        """ VGG19
        """
        model_ft = models.vgg19(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract, num_layers_to_train, dropout2d_rate)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs,num_classes),
            nn.Sigmoid()
            )
        input_size = img_size

    if model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract, num_layers_to_train, 
                                    dropout2d_rate=dropout2d_rate)
        num_ftrs = model_ft.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        model_ft.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs,num_classes),
            nn.Sigmoid()
            )
        input_size = img_size
        


    return model_ft, input_size



#%% CODICE

# chosen_configurations = get_N_HyperparamsConfigs(N=N)
chosen_configurations = get_N_HyperparamsConfigs(N=N) #TODO

for model_name in model_names:
# for model_name in ['resnet50']:


    print(f'-------------MODEL: {model_name} ----------------------------')
    # print(f'Number of dropout layers in the bottleneck: {num_dropouts}')
    for idx,config in enumerate(chosen_configurations): 
        print(f'Starting config {idx}: {config}')
        lr = config[0]
        wd = config[1]
        dropout_rate = config[2]
        batch_size = config[3]
        dropout2d_rate = config[4]
        train_batch_size = batch_size
        test_batch_size = batch_size_valid
        
       
        experiment_run = f'DBT_{model_name}_{strftime("%a_%d_%b_%Y_%H:%M:%S", gmtime())}'
        output_dir = f'./saved_models_baseline/{model_name}/{experiment_run}'
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        
        with open(os.path.join(output_dir,'train_metrics.txt'),'w') as f_out:
            f_out.write('epoch,loss,accuracy\n')
        
        with open(os.path.join(output_dir,'val_metrics.txt'),'w') as f_out:
            f_out.write('epoch,loss,accuracy\n')
            
        with open(os.path.join(output_dir,'run_info.txt'),'w') as f_out:
             f_out.write(run_info_to_be_written)
        
        
        
        # all datasets
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        # train set
        train_dataset = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=True,
            num_workers=workers, pin_memory=False)

        # test set
        test_dataset = datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=True,#False
            num_workers=workers, pin_memory=False)

        # Create training and validation dataloaders
        dataloaders_dict = {
            'train': train_loader,
            'val': test_loader    
            }
        
        
        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        feature_extract = True
        
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #print(f'model_name={model_name[:9]}')
        # Initialize the model for this run
        # model_ft, input_size = initialize_model(actual_model_name, num_classes, feature_extract, dropout_rate, num_dropouts, num_layers_to_train, use_pretrained=True)
        model_ft, input_size = initialize_model(actual_model_name, num_classes, feature_extract, 
                                                dropout_rate, num_layers_to_train, dropout2d_rate=dropout2d_rate, use_pretrained=pretrained)

        with open(os.path.join(output_dir,'model_architecture.txt'),'w') as f_out:
            f_out.write(f'{model_ft}')
 
        # Send the model to GPU
        model_ft = model_ft.to(device)
        
        #TODO MULTI GPU MODEL
        model_ft = torch.nn.DataParallel(model_ft)
        # model_ft = ddp(model_ft)

        
        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if feature_extract:
            params_to_update = []
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t",name)
        else:
            for name,param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        
     
        joint_optimizer_specs = [{'params': params_to_update, 'lr': lr, 'weight_decay': wd}]# bias are now also being regularized
        if optimiser == 'adam':
            optimizer_ft = torch.optim.Adam(joint_optimizer_specs)
        elif optimiser == 'sgd':
            optimizer_ft = torch.optim.SGD(joint_optimizer_specs)
        elif optimiser == 'rms_prop':
            optimizer_ft = torch.optim.RMSprop(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=joint_lr_step_size, gamma=gamma_value)
        
        # Setup the loss fxn
        criterion = nn.BCELoss()
        
        # Train and evaluate
        model_ft, val_accs, train_accs, val_loss, train_loss, best_accuracy= train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"),window=window, patience=patience)
        
        #
        
        num_dropouts=999 #fake number to exclude num_dropouts in further experiments...
        if not os.path.exists(f'./saved_models_baseline/{model_name}/experiments_setup_massBenignMalignant.txt'):
            with open(f'./saved_models_baseline/{model_name}/experiments_setup_massBenignMalignant.txt', 'w') as out_file:
                out_file.write('{experiment_run},{num_layers_to_train},{lr},{wd},{dropout_rate},{dropout2d_rate},{num_dropouts},{batch_size},{best_accuracy}\n')

        with open(f'./saved_models_baseline/{model_name}/experiments_setup_massBenignMalignant.txt', 'a') as out_file: #TODO ricordati di cambiare il nome del txt se cambia esperimento
            out_file.write(f'{experiment_run},{num_layers_to_train},{lr},{wd},{dropout_rate},{dropout2d_rate},{num_dropouts},{batch_size},{best_accuracy}\n')

        
        
        
        # Plots
        val_accs_npy = val_accs
        train_accs_npy = train_accs
        x_axis = range(0,len(val_accs_npy))
        plt.figure()
        plt.plot(x_axis,train_accs_npy,'*-k',label='Training')
        plt.plot(x_axis,val_accs_npy,'*-b',label='Validation')
        # plt.ylim(bottom=0.5,top=1)
        plt.legend()
        b_acc = best_accuracy
        plt.title(f'Accuracy\n{actual_model_name}; BCE Loss; last {num_layers_to_train} conv layers trained\nLR: {lr}, WD: {wd}, dropout: {dropout_rate}, dropout2D: {dropout2d_rate}\nbest val acc: {np.round(b_acc,decimals=2)}, batch size: {batch_size}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid()
        plt.savefig(os.path.join(output_dir,experiment_run+'_acc.pdf'))
        
        
        
        # val_loss_npy = [elem.detach().cpu() for elem in val_loss]
        # train_loss_npy = [elem.detach().cpu() for elem in train_loss]
        x_axis = range(0,len(val_loss))
        plt.figure()
        plt.plot(x_axis,train_loss,'*-k',label='Training')
        plt.plot(x_axis,val_loss,'*-b',label='Validation')
        # plt.ylim(bottom=-0.5)
        plt.legend()
        plt.title(f'Loss\n{actual_model_name}; BCE Loss; last {num_layers_to_train} conv layers trained\nLR: {lr}, WD: {wd}, dropout: {dropout_rate},  dropout2D: {dropout2d_rate}, n_dropout2D: {num_d2d}\nbatch size: {batch_size}, optimiser: {optimiser}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.savefig(os.path.join(output_dir,experiment_run+'_loss.pdf'))
        
        # with open(os.path.join(output_dir,experiment_run+'_accuracies.txt'),'w') as f_out:
        #           f_out.write(train_accs_npy+'\n'+val_accs_npy)
        print(f'End of config {idx}: {config}')

print('All experiments saved!')
