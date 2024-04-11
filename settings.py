experiment_task = 'DBT' #'ADNI' 
# base_architecture = 'vgg19'
base_architecture = 'resnet18' #'resnet50'
#base_architecture = 'densenet121'


img_size = 224 #124# 224 #336
num_classes = 2 
num_prots_per_class = 5 #40 #TODO
num_filters = 128 #512 #256 #128 #256 #TODO
prototype_shape = (num_classes*num_prots_per_class, num_filters, 1, 1) #40 #60 #16 #40
prototype_activation_function = 'log'
# add_on_layers_type = 'regular'
add_on_layers_type = 'bottleneck'

wd = 1e-2 #TODO
num_layers_to_train = 20 #TODO aggiunto da noi
dropout_proportion = 0.4 #0.7 #TODO aggiunto noi

data_path = './datasets/' #
# train_dir = data_path + 'push_augmented/' #
train_dir = data_path + 'push_augmentor/' #
# train_dir = data_path + 'push/' # TODO


# test_dir = data_path + 'valid' #'valid/' #
# test_dir = data_path + 'valid_augmented' #'valid/' #
test_dir = data_path + 'valid_augmented/' #'valid/' #TODO

# train_push_dir = data_path + 'push/' #
train_push_dir = data_path + 'push_augmented/' #

train_batch_size = 10 #20 #40
test_batch_size = 2
train_push_batch_size = 10 #40 #90 #40 #4

joint_optimizer_lrs = {'features': 5e-05, #1e-06,#1e-4 #TODO
                       'add_on_layers': 1e-06, #3e-3,
                       'prototype_vectors': 1e-06} #3e-3}
joint_lr_step_size = 10 #5 #TODO

warm_optimizer_lrs = {'add_on_layers': 5e-05, #3e-3,
                      'prototype_vectors': 5e-05} #3e-3}

last_layer_optimizer_lr = 1e-06 #1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.9, #0.95, #0.8,
    'sep': -0.05, #-0.05, #-0.08,
    'l1': 1e-5
}

num_train_epochs = 100 #42 #TODO era 1000 con EarlyStopping
num_warm_epochs = 5

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

from time import gmtime,strftime
experiment_run = f'{experiment_task}_{strftime("%a_%d_%b_%Y_%H:%M:%S", gmtime())}'
