import os
import shutil
import time


import torch
import torch.utils.data
from torch.utils.data import SubsetRandomSampler
# import torch.utils.data.distributed
# from torch.nn.parallel import DistributedDataParallel as ddp
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re

from helpers import makedir
import model
import push
import prune
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

from stratified_group_data_splitting import StratifiedGroupKFold


import numpy as np
import random
from matplotlib import pyplot as plt
import pandas as pd

torch.cuda.empty_cache()
#%% SEED FIXED TO FOSTER REPRODUCIBILITY
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    start = time.time()
    set_seed(seed=1)
    parser = argparse.ArgumentParser()
    parser.add_argument('gpuid', nargs=1, type=str) #TODO
    # python3 main.py -gpuid=0,1,2,3
    parser.add_argument('runinfo', nargs=1, type=str) #TODO

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0] #TODO
    print(os.environ['CUDA_VISIBLE_DEVICES'])
    
    
    os_env_cudas = os.environ['CUDA_VISIBLE_DEVICES']
    os_env_cudas_splits = os_env_cudas.split(sep=',')
    workers = 4*len(os_env_cudas_splits) #TODO METTERE 4* QUANDO POSSIBILE
    
    
    # book keeping namings and code
    # from settings import base_architecture, img_size, prototype_shape, num_classes, \
    #                      prototype_activation_function, add_on_layers_type, experiment_run, \
    #                          num_prots_per_class, num_filters
    from settings import base_architecture, img_size, num_classes, \
                         prototype_activation_function, add_on_layers_type, \
                         num_filters, experiment_task, num_layers_to_train
    
    # VARIABLES DEFINITION
    # TODO
    lr_features = [1e-06, 1e-07]
    lr_add_on = [1e-06, 1e-07]
    lr_prot_vector = [1e-06, 1e-07]
    wd = [1e-03, 1e-02]
    dropout_proportion = [0.4]
    # batch_size = 999
    train_batch_size = [40, 20]
    clst = [0.8, 0.9, 0.6]
    sep = [-0.08, -0.05, -0.1]
    l1 = [1e-4, 1e-3, 1e-5]
    num_prots_per_class = [5, 20, 40]


    def get_N_HyperparamsConfigs(N=0, lr_features=lr_features, 
                                 lr_add_on=lr_add_on, lr_prot_vector=lr_prot_vector, 
                                 wd=wd, dropout_proportion=dropout_proportion,
                                 train_batch_size=train_batch_size, clst=clst,
                                 sep=sep, l1=l1,
                                 num_prots_per_class=num_prots_per_class):
        configurations = {}
        h = 1
        for i in lr_features:
            for j in lr_add_on:
                for k in lr_prot_vector:
                    for l in wd:
                        for m in dropout_proportion:
                            for n in train_batch_size:
                                for o in clst:
                                    for p in sep:
                                        for q in l1:
                                            for r in num_prots_per_class:
                                                configurations[f'config_{h}'] = [i, j, k, l, m, n, o, p, q, r]
                                                h += 1
                
                         
        configurations_key = list(configurations.keys())
        chosen_configs = sorted(random.sample(configurations_key,N)) 
        return [configurations[x] for x in chosen_configs]



    
    # Configurations
    # N = len(lr_features) * len(lr_add_on) * len(lr_prot_vector) * len(wd) * len(dropout_proportion) * len(train_batch_size) * len(clst) * len(sep) * len(l1) * len(num_prots_per_class)
    
    N = 30
    chosen_configurations = get_N_HyperparamsConfigs(N=N)
    
    for idx, config in enumerate(chosen_configurations[20:23]): # TODO
        
        # #TODO
        # temp = [23, 24, 25, 26]
        # idx = temp[idx]
        # #
        # if idx == 25:
        #     continue
        idx += 20
        
        experiment_run = f'{experiment_task}_{time.strftime("%a_%d_%b_%Y_%H:%M:%S", time.gmtime())}_config{idx}'
        lr_features = config[0]
        lr_add_on = config[1]
        lr_prot_vector = config[2]
        wd = config[3] 
        dropout_proportion = config[4]
        train_batch_size = config[5]
        clst = config[6]
        sep = config[7]
        l1 = config[8]
        num_prots_per_class = config[9]
        
        prototype_shape = (num_classes*num_prots_per_class, num_filters, 1, 1)
        base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
        
        model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
        makedir(model_dir)
        #TODO scrittura di un file di informazioni sulla run in oggetto
        with open(os.path.join(model_dir,'run_info.txt'),'w') as fout:
            fout.write(f'{args.runinfo}')
        #
        with open(os.path.join(model_dir,'configuratrion_params.txt'),'w') as fout:
            fout.write(f'lr_features={lr_features}\nlr_add_on={lr_add_on}\nlr_prot_vector={lr_prot_vector}\nwd={wd}\ndropout_proportion={dropout_proportion}\ntrain_batch_size={train_batch_size}\nclst={clst}\nsep={sep}\nl1={l1}\nnum_prots_per_class={num_prots_per_class}')
        #
        #
        shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
        shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
        
        log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
        # img_dir = os.path.join(model_dir, 'img')
        # makedir(img_dir)
        weight_matrix_filename = 'outputL_weights'
        prototype_img_filename_prefix = 'prototype-img'
        prototype_self_act_filename_prefix = 'prototype-self-act'
        proto_bound_boxes_filename_prefix = 'bb'
        
        
        log(f'configuration {idx}: {config}')

        # load the data
        from settings import original_dir, augm_dir, \
                             test_batch_size, train_push_batch_size
        
        
        # TODO metterei batch size in configurations
        # from model import dropout_proportion # aggiunta noi questa variabile SBAGLIATO, STA IN SETTINGS
        # TODO SISTEMARE
        from settings import data_path
        path_to_csv = os.path.join(data_path, 'push_e_valid_MLO', 'push_e_valid_MLO_bm.csv')
        path_to_csv_augmented = os.path.join(augm_dir, 'push_e_valid_MLO_augmented_bm.csv')
        df_original = pd.read_csv(path_to_csv, sep=',', index_col='file_name')
        df_augmented = pd.read_csv(path_to_csv_augmented, sep=',', index_col='file_name')
    
        
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        
        # Here train_dataset becomes the img_original_dataset and push_dataset becomes the img_augmented_dataset
        # all datasets
        img_original_dataset = datasets.ImageFolder(
            original_dir,
            transforms.Compose([
                transforms.Grayscale(num_output_channels=3), #TODO
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
    
        img_original_dataset_push = datasets.ImageFolder(
            original_dir,
            transforms.Compose([
                transforms.Grayscale(num_output_channels=3), #TODO
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
            ]))
        
        img_augmented_dataset = datasets.ImageFolder(
            augm_dir,
            transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
        # train set
        # train_dataset = datasets.ImageFolder(
        #     train_dir,
        #     transforms.Compose([
        #         transforms.Grayscale(num_output_channels=3), #TODO
        #         transforms.Resize(size=(img_size, img_size)),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset, batch_size=train_batch_size, shuffle=True,
        #     num_workers=workers, pin_memory=False) #TODO cambiare num_workers=4*num_gpu
        # # push set
        # train_push_dataset = datasets.ImageFolder(
        #     train_push_dir,
        #     transforms.Compose([
        #         transforms.Grayscale(num_output_channels=3),
        #         transforms.Resize(size=(img_size, img_size)),
        #         transforms.ToTensor(),
        #     ]))
        # train_push_loader = torch.utils.data.DataLoader(
        #     train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        #     num_workers=workers, pin_memory=False)
        # # test set
        # test_dataset = datasets.ImageFolder(
        #     test_dir,
        #     transforms.Compose([
        #         transforms.Grayscale(num_output_channels=3),
        #         transforms.Resize(size=(img_size, img_size)),
        #         transforms.ToTensor(),
        #         normalize,
        #     ]))
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset, batch_size=test_batch_size, shuffle=True, #TODO messo True, era falso
        #     num_workers=workers, pin_memory=False)
        
        # we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
        # log('training set size: {0}'.format(len(train_loader.dataset)))
        # log('push set size: {0}'.format(len(train_push_loader.dataset)))
        # log('test set size: {0}'.format(len(test_loader.dataset)))
        log('batch size: {0}'.format(train_batch_size))
        
        #%% construct the model

            
        # define optimizer
        from settings import joint_optimizer_lrs, joint_lr_step_size
        joint_optimizer_lrs['features'] = lr_features
        joint_optimizer_lrs['add_on_layers'] = lr_add_on 
        joint_optimizer_lrs['prototype_vectors'] = lr_prot_vector
 
        # weighting of different training losses
        from settings import coefs
        coefs['clst'] = clst
        coefs['sep'] = sep
        coefs['l1'] = l1
        
        # number of training epochs, number of warm epochs, push start epoch, push epochs
        from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs
        
        ################
        # start k-fold #
        ################
        
        # Separate train and valid_augmented
        X = np.array(df_augmented.index) # is an array of names
        group = np.array(df_augmented['patient_id'])
        y = np.array(df_augmented['label'])
        
        y = np.array([0 if elem=='benign' else 1 for elem in y]) #TODO modificare con la stringa opportuna
        
        
        y_original = np.array(df_original['label'])
        y_original = np.array([0 if elem=='benign' else 1 for elem in y_original]) #TODO modificare con la stringa opportuna

        # create the dictionary containing dict = {'img_name': 'img_index'} for the dataset
        dict_augmented = {key[0]:value for value, key in enumerate(img_augmented_dataset.imgs)}
        dict_original = {key[0]:value for value, key in enumerate(img_original_dataset.imgs)}
        
        best_val_accuracy_folds = []
        
        k = 5
        
        x = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42) #reproducibility, 5-fold
        for fold, (train_idx, valid_augm_idx) in enumerate(x.split(X,y,groups=group)):
            
            print(f'Proporzione di maligni su totale TRAIN: {np.sum(y[train_idx])/len(y[train_idx])}')
            print(f'Proporzione di maligni su totale VALID_AUGM: {np.sum(y[valid_augm_idx])/len(y[valid_augm_idx])}')

            start_fold = time.time()
            
            ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                          pretrained=True, img_size=img_size,
                                          prototype_shape=prototype_shape,
                                          num_classes=num_classes, 
                                          dropout_proportion=dropout_proportion,
                                          prototype_activation_function=prototype_activation_function,
                                          add_on_layers_type=add_on_layers_type)
            
            log('CURRENT INPUT CHANNELS: {0}'.format(ppnet.current_in_channels))  
            
            
            #if prototype_activation_function == 'linear':
            #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
            
            with open(os.path.join(model_dir,'architecture.txt'),'w') as fout:
                fout.write(f'{ppnet}\n')
                at_least_one_sequential = False
                for m in ppnet.features.modules():
                    if isinstance(m,torch.nn.Sequential):
                        at_least_one_sequential = True
                        fout.write(f'{m}\n')
                    if not at_least_one_sequential:
                        fout.write(f'{m}\n')
                        
                for m in ppnet.add_on_layers.modules():
                    if isinstance(m,torch.nn.Sequential):
                        fout.write(f'{m}\n')
                    
                # for m in ppnet.prototype_vectors.modules():
                #     fout.write(f'{m}\n')
                    
                for m in ppnet.last_layer.modules():
                    fout.write(f'{m}\n')
                
            ppnet = ppnet.cuda()
            ppnet_multi = torch.nn.DataParallel(ppnet)
            class_specific = True
            
            joint_optimizer_specs = \
            [{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': wd}, # bias are now also being regularized
             {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': wd},
             {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
            ]
            joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
            # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)
            
            from settings import warm_optimizer_lrs
            warm_optimizer_specs = \
            [{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': wd},
             {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
            ]
            warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
            
            from settings import last_layer_optimizer_lr
            last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
            last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
            
            
            out_dir_fold = os.path.join(model_dir, f'fold{fold}')
            
            if not os.path.exists(out_dir_fold):
                os.makedirs(out_dir_fold)
            
            log2, logclose2 = create_logger(log_filename=os.path.join(out_dir_fold, f'train{fold}.log'))
            log2(f'Starting FOLD {fold}/{k} of CONFIGURATION {idx}/{len(chosen_configurations)}')
            
            img_dir = os.path.join(out_dir_fold, 'img')
            makedir(img_dir)
            
            # train_names e valid_augm_names sono nomi presi da csv, quindi iniziano con ./datasets/...
           
            # dato il nome, vedere l'indice nel dataset FATTO (penso)
            # train_names get prefix -> unique, get those names for push from img_original_dataset
            #prefix = os.getcwd()
        
            # train_idx = [dict_augmented[os.path.join(prefix, name[1:])] for name in train_names]
            train_idx = [dict_augmented[X[index]] for index in train_idx]
            
            train_sampler = SubsetRandomSampler(train_idx)
            train_loader = torch.utils.data.DataLoader(img_augmented_dataset,
                                                       batch_size=train_batch_size,
                                                       #shuffle=True,
                                                       num_workers=workers,
                                                       pin_memory=False,
                                                       sampler=train_sampler)
            
            # get the names of the non-augmented images to create the push and valid datasets
            sep = 'png'
            
            # name_prefixes_push = [f'{X[index].split(sep=sep)[0]}png' for index in train_idx]
            original_prefix = list(dict_original.keys())[0]
            original_prefix = original_prefix.split(os.sep)
            original_prefix = os.path.join(*original_prefix[:-2])
            name_prefixes_push = []
            for index in train_idx:
                img_augm_prefix = f'{X[index].split(sep=sep)[0]}png'
                img_augm_prefix = img_augm_prefix.split(os.sep)
                tail = img_augm_prefix[-1]
                tail_splits = tail.split(sep='Mass')
                img_augm_prefix[-1] = f'Mass{tail_splits[-1]}'
                img_augm_prefix = os.path.join(*img_augm_prefix[-2:])
                img_augm_prefix = os.path.join(original_prefix, img_augm_prefix)
                name_prefixes_push.append(img_augm_prefix)
            push_names = list(set(name_prefixes_push))
         
            name_prefixes_valid = []
            for index in valid_augm_idx:
                img_augm_prefix = f'{X[index].split(sep=sep)[0]}png'
                img_augm_prefix = img_augm_prefix.split(os.sep)
                tail = img_augm_prefix[-1]
                tail_splits = tail.split(sep='Mass')
                img_augm_prefix[-1] = f'Mass{tail_splits[-1]}'
                img_augm_prefix = os.path.join(*img_augm_prefix[-2:])
                img_augm_prefix = os.path.join(original_prefix, img_augm_prefix)
                name_prefixes_valid.append(img_augm_prefix)
            valid_names = list(set(name_prefixes_valid))
            
            # get the corresponding indices in the img_original_dataset
            # push_idx = [dict_original[os.path.join(prefix, name[1:])] for name in push_names]
            # valid_idx = [dict_original[os.path.join(prefix, name[1:])] for name in valid_names]
            push_idx = [dict_original[name] for name in push_names]
            valid_idx = [dict_original[name] for name in valid_names]
     
            # using those indices, create the push and valid dataloader
            push_sampler = SubsetRandomSampler(push_idx)
            train_push_loader = torch.utils.data.DataLoader(img_original_dataset_push,
                                                            batch_size=train_push_batch_size,
                                                            shuffle=False,
                                                            num_workers=workers,
                                                            pin_memory=False,
                                                            sampler=push_sampler)
           
            valid_sampler = SubsetRandomSampler(valid_idx)
            test_loader = torch.utils.data.DataLoader(img_original_dataset,
                                                             batch_size=test_batch_size,
                                                             #shuffle=True, #TODO messo True, era falso
                                                             num_workers=workers,
                                                             pin_memory=False,
                                                             sampler=valid_sampler)  
            #
            
            # train the model
            log2('start training')
            # import copy
            
            #
            accs_noiter_valid = []
            losses_noiter_valid = []
            accs_noiter_train = []
            losses_noiter_train = []
            triggered_count = 0
            earlystopped_acc = 0
            best_acc = 0
            first_time = True
            #
            for epoch in range(num_train_epochs):
                log2('epoch: \t{0}'.format(epoch))
                
                
        
                if epoch < num_warm_epochs:
                    tnt.warm_only(model=ppnet_multi, log=log2)
                    accu_train,loss_train = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                                  class_specific=class_specific, coefs=coefs, log=log2)
                else:
                    tnt.joint(model=ppnet_multi, num_layers_to_train=num_layers_to_train, log=log2)
                    if first_time:
                        first_time = False
                        params_to_update = ppnet_multi.parameters()
                        log2("\tParams to learn:")
                        params_to_update = []
                        for name,param in ppnet_multi.named_parameters():
                            if param.requires_grad == True:
                                params_to_update.append(param)
                                log2(f"\t\t{name}")
                        
                    # joint_lr_scheduler.step()
                    accu_train,loss_train = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                                  class_specific=class_specific, coefs=coefs, log=log2)
            
                accu, loss = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log2)
                
                accs_noiter_valid.append(accu)
                losses_noiter_valid.append(loss)
                accs_noiter_train.append(accu_train)
                losses_noiter_train.append(loss_train)
                
                # save.save_model_w_condition(model=ppnet, model_dir=out_dir_fold, model_name=str(epoch) + 'nopush', accu=accu,
                                            # target_accu=0.68, log=log2)
            
                if epoch >= push_start and epoch in push_epochs:
                    
                    loss_npy = np.array(losses_noiter_valid)
                    window=5 #TODO
                    ultimo = np.mean(loss_npy[-window:])
                    penultimo = np.mean(loss_npy[-2*window:-window])
                    
                    if ultimo-penultimo >= 0:
                        triggered_count+=1 #set #TODO usa questa linea non quella sotto
                    elif ultimo-penultimo < 0:
                        #reset
                        triggered_count=0 
                   
                    if triggered_count==4:
                        # Pazienza di un push da quando si triggera la prima volta
                        # a quando esco. Così, l'ultima cartella di immagini salvate
                        # è proprio quella relativa al push triggerante la prima volta.
                        log2(f'Early stopping at epoch {epoch}-------------------------')
                        break
                                
                    push.push_prototypes(
                        train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                        prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
                        class_specific=class_specific,
                        preprocess_input_function=preprocess_input_function, # normalize if needed
                        prototype_layer_stride=1,
                        root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                        epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                        prototype_img_filename_prefix=prototype_img_filename_prefix,
                        prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                        proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                        save_prototype_class_identity=True,
                        log=log2)
                    
                    accu,loss = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log2)
                    
                    # SE ERA UNA EPOCA DI PUSH, ELIMINIAMO IL VALORE PRE-PUSH
                    accs_noiter_valid.pop()
                    losses_noiter_valid.pop()                    
                    #
                    accs_noiter_valid.append(accu)
                    losses_noiter_valid.append(loss)
                    # # Copiamo lo stesso valore del training per la push epoch:
                    # accs_noiter_train.append(accu_train)
                    # losses_noiter_train.append(loss_train)  
                    
                    if triggered_count==1:
                        earlystopped_acc = accu  
                        best_acc = earlystopped_acc
                        save.save_model_w_condition(model=ppnet, model_dir=out_dir_fold, model_name=str(epoch) + 'push', accu=earlystopped_acc,
                                                    target_accu=0.50, log=log2)
                    
                    
                    
                    if prototype_activation_function != 'linear':
                        tnt.last_only(model=ppnet_multi, log=log2)
                        for i in range(20):
                            log2('iteration: \t{0}'.format(i))
                            _,_ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                          class_specific=class_specific, coefs=coefs, log=log2)
                            accu,_ = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                            class_specific=class_specific, log=log2)
                            if accu > best_acc:
                                best_acc = accu
                                save.save_model_w_condition(model=ppnet, model_dir=out_dir_fold, model_name=str(epoch) + '_' + str(i) + 'push', accu=best_acc,
                                                        target_accu=0.50, log=log2)
               
            logclose2()
            stop = time.time()
            
            run_time = np.round((stop - start_fold) / 60, decimals=1)
            coeff_crs_ent = coefs['crs_ent']
            coeff_clst = coefs['clst']
            coeff_sep = coefs['sep']
            
            accs_noiter_train_npy = np.array(accs_noiter_train)
            np.save(os.path.join(out_dir_fold, f'npy_accs_noiter_train_fold{fold}.npy'), 
                    accs_noiter_train_npy)
        
            accs_noiter_valid_npy = np.array(accs_noiter_valid)
            np.save(os.path.join(out_dir_fold, f'npy_accs_noiter_valid_fold{fold}.npy'), 
                    accs_noiter_valid_npy)
        
            losses_noiter_train_npy = np.array(losses_noiter_train)
            np.save(os.path.join(out_dir_fold, f'npy_loss_noiter_train_fold{fold}.npy'),
                    losses_noiter_train_npy)
        
            losses_noiter_valid_npy = np.array(losses_noiter_valid)
            np.save(os.path.join(out_dir_fold, f'npy_loss_noiter_valid_fold{fold}.npy'),
                    losses_noiter_valid_npy)
        
            
            x_axis = range(0,len(accs_noiter_valid))
            plt.figure()
            plt.plot(x_axis, accs_noiter_train,'*-k',label='Training')
            plt.plot(x_axis,accs_noiter_valid,'*-b',label='Validation')
            # plt.ylim(bottom=0.5,top=1)
            plt.legend()
            # plt.title(f'Accuracy {base_architecture}, earlyStAcc:{np.round(earlystopped_acc, decimals=2)}, bestAcc:{np.round(best_acc, decimals=2)}, imgSize:{img_size}, protPerClass:{num_prots_per_class}, numFilters:{num_filters}\ndropout:{dropout_proportion}, trainBSize:{train_batch_size}, testBSize:{test_batch_size}, pushBSize:{train_push_batch_size}, CE:{coeff_crs_ent}, CLS:{coeff_clst}, SEP:{coeff_sep}')
            plt.title(f'Accuracy, ESAcc:{np.round(earlystopped_acc, decimals=2)}, BAcc:{np.round(best_acc, decimals=2)}, imSize:{img_size}, Prots:{num_prots_per_class}, Filters:{num_filters}\ndrop:{dropout_proportion}, TrBSize:{train_batch_size}, TeBSize:{test_batch_size}, PBSize:{train_push_batch_size}, CE:{coeff_crs_ent}, CLS:{coeff_clst}, SEP:{coeff_sep}')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.grid()
            plt.savefig(os.path.join(out_dir_fold, 'acc_noiter.pdf'))
            
            plt.figure()
            plt.plot(x_axis, losses_noiter_train,'*-k',label='Training')
            plt.plot(x_axis,losses_noiter_valid,'*-b',label='Validation')
            # plt.ylim(bottom=0.5,top=1)
            plt.legend()
            # plt.title(f'Loss {base_architecture}\nimgSize:{img_size}, protPerClass:{num_prots_per_class}, numFilters:{num_filters}, dropout:{dropout_proportion}\ntrainBSize:{train_batch_size}, testBSize:{test_batch_size}, pushBSize:{train_push_batch_size}\nCE:{coeff_crs_ent}, CLS:{coeff_clst}, SEP:{coeff_sep}')
            plt.title(f'Loss, imSize:{img_size}, Prots:{num_prots_per_class}, Filters:{num_filters}\ndrop:{dropout_proportion}, TrBSize:{train_batch_size}, TeBSize:{test_batch_size}, PBSize:{train_push_batch_size}, CE:{coeff_crs_ent}, CLS:{coeff_clst}, SEP:{coeff_sep}')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid()
            plt.savefig(os.path.join(out_dir_fold,'loss_noiter.pdf'))
        
            
            # with open('./saved_models/experiments_setup.txt', 'a') as out_file:
            #     out_file.write(f'{experiment_run},{base_architecture},{img_size},{num_classes},{num_prots_per_class},{num_filters},{train_batch_size},{test_batch_size},{train_push_batch_size},{coeff_crs_ent},{coeff_clst},{coeff_sep},{num_warm_epochs},{num_train_epochs},{dropout_proportion},{run_time}\n')
            with open(os.path.join(out_dir_fold, 'fold_performance.txt'), 'w') as out_file:
                out_file.write(f'Ended Fold {fold+1}/{k}. Best validation accuracy = {np.round(best_acc, decimals=3)}')
   
            best_val_accuracy_folds.append(best_acc)
            
        best_val_accuracy_folds_npy = np.array(best_val_accuracy_folds)
        mean_val_accuracy_configuration = np.mean(best_val_accuracy_folds_npy)
        with open(os.path.join(model_dir,'configuratrion_params.txt'),'a') as fout:
            fout.write('\n')
            fout.write(f'mean_val_acc={mean_val_accuracy_configuration}\n')
            fout.write(f'Folds_val_accu={best_val_accuracy_folds_npy}\n')
        np.save(os.path.join(model_dir, 'mean_val_accuracy.npy'), mean_val_accuracy_configuration)
        logclose()
        
if __name__ == '__main__':
    main()