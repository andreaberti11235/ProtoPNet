##### MODEL AND DATA LOADING
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image
import glob
import seaborn as sn
import shutil
from sklearn.metrics import confusion_matrix, \
                            accuracy_score, \
                            balanced_accuracy_score, \
                            precision_score, \
                            recall_score, \
                            roc_auc_score, \
                            f1_score, \
                            fbeta_score
                            

'''
precision: TP/(TP+FP) avere pochi falsi positivi, in questo caso è meno importante rispetto alla recall --> masse benigne che vengono comunque mandate a biopsia

Note that in binary classification:
    recall of the positive class TP/(TP+FN) is “sensitivity”;
    recall of the negative class TN/(TN+FP) is “specificity”.
recall (SENSITIVITY): TP/(TP+FN) avere pochi falsi negativi, quando è positivo lo becchi bene --> masse maligne che non vengono rilevate e potenzialmente escluse da biopsia: è molto pericoloso


ROC: x-axis FP rate (FP/TN+FP), y-axis TP rate (RECALL or SENSITIVITY)
'''

import re

import os
import copy

from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('test_dir', help='Path to the directory of the test set (parent containing benign/ and malignant/')
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-model', nargs=1, type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# specify the test image to be analyzed
test_dir = args.test_dir


load_model_path = args.model[0]
load_model_dir = os.path.dirname(load_model_path)
load_model_name = os.path.basename(load_model_path)

sub_folders = ['predette_bene', 'predette male']
sub_sub_folders = ['benign', 'malignant']

for sub_folder in sub_folders:
    for sub_sub_folder in sub_sub_folders:
        new_folder = os.path.join(test_dir, load_model_name[:-4], sub_folder, sub_sub_folder)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

y = []


for folder in ['benign', 'malignant']:
    img_dir = os.path.join(test_dir, folder)
    if folder == 'benign':
        real_label = 0
    if folder == 'cancer':
    #if folder == 'malignant':
        real_label = 1
    
    for img in tqdm(glob.glob(os.path.join(img_dir, '*.png'))):
        
        
        # load the model
        # check_test_accu = True

        #if load_model_dir[-1] == '/':
        #    model_base_architecture = load_model_dir.split('/')[-3]
        #    experiment_run = load_model_dir.split('/')[-2]
        #else:
        #    model_base_architecture = load_model_dir.split('/')[-2]
        #    experiment_run = load_model_dir.split('/')[-1]
        
        # model_base_architecture = load_model_dir.split('/')[2]
        # experiment_run = '/'.join(load_model_dir.split('/')[3:])
        
        # save_analysis_path = os.path.join(test_image_dir, model_base_architecture,
        #                                   experiment_run, load_model_name)
        # makedir(save_analysis_path)
                

        epoch_number_str = re.search(r'\d+', load_model_name).group(0)
        start_epoch_number = int(epoch_number_str)
        
        # log('model base architecture: ' + model_base_architecture)
        # log('experiment run: ' + experiment_run)
        
        ppnet = torch.load(load_model_path)
        ppnet = ppnet.cuda()
        ppnet_multi = torch.nn.DataParallel(ppnet)
        
        img_size = ppnet_multi.module.img_size
        prototype_shape = ppnet.prototype_shape
        max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]
        
        class_specific = True
        
        # load the test image and forward it through the network
        
        normalize = transforms.Normalize(mean=mean,
                                         std=std)
        
        preprocess = transforms.Compose([
           # transforms.Resize((img_size,img_size)),
           # transforms.ToTensor(),
           # normalize
           
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])
        
        img_name = os.path.basename(img)
        
        # img_pil = Image.open(img).convert('RGB') #TODO dobbiamo farla diventare una finta rgb
        img_pil = Image.open(img)

        img_tensor = preprocess(img_pil)
        img_variable = Variable(img_tensor.unsqueeze(0))
        
        images_test = img_variable.cuda()
        labels_test = torch.tensor([real_label])
        
        logits, min_distances = ppnet_multi(images_test)
        conv_output, distances = ppnet.push_forward(images_test)
        prototype_activations = ppnet.distance_2_similarity(min_distances)
        prototype_activation_patterns = ppnet.distance_2_similarity(distances)
        if ppnet.prototype_activation_function == 'linear':
            prototype_activations = prototype_activations + max_dist
            prototype_activation_patterns = prototype_activation_patterns + max_dist
        
        
    
        # for i in range(logits.size(0)):
        #     y.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item(), torch.softmax(logits, dim=1)[i].item()))
        
        y.append((torch.argmax(logits).item(), labels_test.item(), torch.softmax(logits, dim=1)[0][1].item()))

        predicted_cls = y[-1][0]
        correct_cls = y[-1][1]
        
        out_parent = os.path.join(test_dir, load_model_name[:-4])
        
        if predicted_cls == correct_cls:
            shutil.copy(img, os.path.join(out_parent, 'predette_bene', folder, img_name))
        
        else:
            shutil.copy(img, os.path.join(out_parent, 'predette male', folder, img_name))

classes = ('Benign','Malignant') 
y_pred = [elem[0] for elem in y]
y_true = [elem[1] for elem in y]
y_score = [elem[2] for elem in y] #da usarsi in ROC

# metrics
metrics_acc = accuracy_score(y_true,y_pred)
metrics_bal_acc = balanced_accuracy_score(y_true,y_pred)
metrics_precision = precision_score(y_true,y_pred)
metrics_recall = recall_score(y_true,y_pred)
metrics_specificity = recall_score(y_true, y_pred, pos_label=0)
metrics_f1score = f1_score(y_true,y_pred)
metrics_f2score = fbeta_score(y_true, y_pred, beta=2)
metrics_auroc = roc_auc_score(y_true,y_score)


with open(os.path.join(out_parent,'metrics.txt'),'w') as fout:
    fout.write('acc,bal_acc,precision,recall,specificity,f1score,f2score,auroc_malignant\n')
    fout.write(f'{metrics_acc},{metrics_bal_acc},{metrics_precision},{metrics_recall},{metrics_specificity},{metrics_f1score},{metrics_f2score},{metrics_auroc}')

cf_mat_norm = confusion_matrix(y_true, y_pred, normalize='true')
cf_mat = confusion_matrix(y_true, y_pred).astype(int)
np.save(os.path.join(out_parent,'confusion_matrix_norm.npy'),cf_mat_norm)
np.save(os.path.join(out_parent,'confusion_matrix.npy'),cf_mat)

df = pd.DataFrame(cf_mat, index = [i for i in classes],
                  columns = [i for i in classes])
plt.figure()
sn.heatmap(df, annot=True, linewidths=.5, cmap='Blues', linecolor='black', fmt="d", vmin=0)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig(os.path.join(out_parent,'confusion_matrix.pdf'),bbox_inches='tight')

df_norm = pd.DataFrame(cf_mat_norm, index=[i for i in classes], columns=[i for i in classes])
plt.figure()
sn.heatmap(df_norm, annot=True, linewidths=.5, cmap='Blues', linecolor='black', vmin=0, vmax=1)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig(os.path.join(out_parent,'confusion_matrix_norm.pdf'),bbox_inches='tight')

