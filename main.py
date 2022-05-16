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
    from settings import base_architecture, img_size, prototype_shape, num_classes, \
                         prototype_activation_function, add_on_layers_type, experiment_run, \
                             num_prots_per_class, num_filters
    
    base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
    
    model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
    makedir(model_dir)
    ##TODO scrittura di un file di informazioni sulla run in oggetto
    with open(os.path.join(model_dir,'run_info.txt'),'w') as fout:
        fout.write(f'{args.runinfo}')
    #
    shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
    shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)
    
    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    img_dir = os.path.join(model_dir, 'img')
    makedir(img_dir)
    weight_matrix_filename = 'outputL_weights'
    prototype_img_filename_prefix = 'prototype-img'
    prototype_self_act_filename_prefix = 'prototype-self-act'
    proto_bound_boxes_filename_prefix = 'bb'
    
    # load the data
    from settings import train_dir, test_dir, train_push_dir, \
                         train_batch_size, test_batch_size, train_push_batch_size
    
    from model import dropout_proportion #TODO aggiunta noi questa variabile
    
    path_to_csv = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/datasets/push_e_valid_MLO/push_e_valid_MLO_bm.csv'
    path_to_csv_augmented = '/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/ppnet_cluster/datasets/push_e_valid_MLO_augmented/push_e_valid_MLO_augmented_bm.csv'
    df_original = pd.read_csv(path_to_csv, sep=',', index_col='file_name')
    df_augmented = pd.read_csv(path_to_csv_augmented, sep=',', index_col='file_name')

    
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    
    # Here train_dataset becomes the img_original_dataset and push_dataset becomes the img_augmented_dataset
    # all datasets
    img_original_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Grayscale(num_output_channels=3), #TODO
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=train_batch_size, shuffle=True,
    #     num_workers=workers, pin_memory=False) #TODO cambiare num_workers=4*num_gpu
    # push set
    img_augmented_dataset = datasets.ImageFolder(
        train_push_dir,
        transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    # train_push_loader = torch.utils.data.DataLoader(
    #     train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    #     num_workers=workers, pin_memory=False)
    # test set
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    # test_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=test_batch_size, shuffle=True, #TODO messo True, era falso
    #     num_workers=workers, pin_memory=False)
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
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=prototype_shape,
                                  num_classes=num_classes,
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
    

        
    # define optimizer
    from settings import joint_optimizer_lrs, joint_lr_step_size, wd
    
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
    
    # weighting of different training losses
    from settings import coefs
    
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
    
    # create the dictionary containing dict = {'img_name': 'img_index'} for the dataset
    dict_augmented = {key[0]:value for value, key in enumerate(img_augmented_dataset.imgs)}
    dict_original = {key[0]:value for value, key in enumerate(img_original_dataset.imgs)}

    
    x = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) #reproducibility, 5-fold
    for fold, (train_names, valid_augm_names) in enumerate(x.split(X,y,groups=group)):
       # train_names e valid_augm_names sono nomi presi da csv, quindi iniziano con ./datasets/...
       # a quanto pare stasera nel dataset invece prende i path assoluti come nomi, quindi bisogna sostituire /home/andrea.berti/a_e_g/ProtoPNet al . che sta all'inizio della stringa
      
       # dato il nome, vedere l'indice nel dataset FATTO (penso)
       # train_names get prefix -> unique, get those names for push from img_original_dataset
       train_idx = [dict_augmented[f'/home/andrea.berti/a_e_g/ProtoPNet{name[1:]}'] for name in train_names] ....controllare
       # valid_augm_idx = [dict_augmented[name] for name in valid_augm_names]
       
       train_sampler = SubsetRandomSampler(train_idx)
       # valid_augm_sampler = SubsetRandomSampler(valid_augm_idx) # questo probabilmente non serve!
       train_loader = torch.utils.data.DataLoader(img_augmented_dataset, batch_size=train_batch_size, sampler=train_sampler)
       
       name_prefixes_push = [name[:100] for name in train_names]
       push_names = unique(name_prefixes_push)
       
       name_prefixes_valid = [name[:100] for name in train_names]
       valid_names = unique(name_prefixes_valid)
    #
    
    # train the model
    log('start training')
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
        log('epoch: \t{0}'.format(epoch))
        
        

        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            accu_train,loss_train = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            if first_time:
                first_time = False
                params_to_update = ppnet_multi.parameters()
                log("\tParams to learn:")
                params_to_update = []
                for name,param in ppnet_multi.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        log(f"\t\t{name}")
                
            # joint_lr_scheduler.step()
            accu_train,loss_train = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)
    
        accu, loss = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        
        accs_noiter_valid.append(accu)
        losses_noiter_valid.append(loss)
        accs_noiter_train.append(accu_train)
        losses_noiter_train.append(loss_train)
        
        # save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
                                    # target_accu=0.68, log=log)
    
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
                log(f'Early stopping at epoch {epoch}-------------------------')
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
                log=log)
            
            accu,loss = tnt.test(model=ppnet_multi, dataloader=test_loader,
                            class_specific=class_specific, log=log)
            
            accs_noiter_valid.append(accu)
            losses_noiter_valid.append(loss)
            # Copiamo lo stesso valore del training per la push epoch:
            accs_noiter_train.append(accu_train)
            losses_noiter_train.append(loss_train)  
            
            if triggered_count==1:
                earlystopped_acc = accu  
                best_acc = earlystopped_acc
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=earlystopped_acc,
                                            target_accu=0.50, log=log)
            
            
            
            if prototype_activation_function != 'linear':
                tnt.last_only(model=ppnet_multi, log=log)
                for i in range(20):
                    log('iteration: \t{0}'.format(i))
                    _,_ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                  class_specific=class_specific, coefs=coefs, log=log)
                    accu,_ = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                    class_specific=class_specific, log=log)
                    if accu > best_acc:
                        best_acc = accu
                        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=best_acc,
                                                target_accu=0.50, log=log)
       
    logclose()
    stop = time.time()
    
    run_time = np.round((stop - start) / 60, decimals=1)
    coeff_crs_ent = coefs['crs_ent']
    coeff_clst = coefs['clst']
    coeff_sep = coefs['sep']
    
    accs_noiter_train_npy = np.array(accs_noiter_train)
    np.save(os.path.join(model_dir,'npy_accs_noiter_train.npy'),accs_noiter_train_npy)

    accs_noiter_valid_npy = np.array(accs_noiter_valid)
    np.save(os.path.join(model_dir,'npy_accs_noiter_valid.npy'),accs_noiter_valid_npy)

    losses_noiter_train_npy = np.array(losses_noiter_train)
    np.save(os.path.join(model_dir,'npy_loss_noiter_train.npy'),losses_noiter_train_npy)

    losses_noiter_valid_npy = np.array(losses_noiter_valid)
    np.save(os.path.join(model_dir,'npy_loss_noiter_valid.npy'),losses_noiter_valid_npy)

    
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
    plt.savefig(os.path.join(model_dir,'acc_noiter.pdf'))
    
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
    plt.savefig(os.path.join(model_dir,'loss_noiter.pdf'))

    
    with open('./saved_models/experiments_setup.txt', 'a') as out_file:
        out_file.write(f'{experiment_run},{base_architecture},{img_size},{num_classes},{num_prots_per_class},{num_filters},{train_batch_size},{test_batch_size},{train_push_batch_size},{coeff_crs_ent},{coeff_clst},{coeff_sep},{num_warm_epochs},{num_train_epochs},{dropout_proportion},{run_time}\n')
    
if __name__ == '__main__':
    main()