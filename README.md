# On the Applicability of Prototypical Part Learning in Medical Images: Breast Masses Classification Using ProtoPNet

This project is the codebase of our paper [On the Applicability of Prototypical Part Learning in Medical Images: Breast Masses Classification Using ProtoPNet](https://doi.org/10.1007/978-3-031-37660-3_38) presented at the International Conference of Pattern Recognition (ICPR) 2022's workshop "Artificial Intelligence for Healthcare Applications" (AIHA).

G. Carloni and A. Berti—These authors share the first authorship.

## Abstract

Deep learning models have become state-of-the-art in many areas, ranging from computer vision to agriculture research. However, concerns have been raised with respect to the transparency of their decisions, especially in the image domain. In this regard, Explainable Artificial Intelligence has been gaining popularity in recent years. The ProtoPNet model, which breaks down an image into prototypes and uses evidence gathered from the prototypes to classify an image, represents an appealing approach. Still, questions regarding its effectiveness arise when the application domain changes from real-world natural images to gray-scale medical images. This work explores the applicability of prototypical part learning in medical imaging by experimenting with ProtoPNet on a breast masses classification task. The two considered aspects were the classification capabilities and the validity of explanations. We looked for the optimal model’s hyperparameter configuration via a random search. We trained the model in a five-fold CV supervised framework, with mammogram images cropped around the lesions and ground-truth labels of benign/malignant masses. Then, we compared the performance metrics of ProtoPNet to that of the corresponding base architecture, which was ResNet18, trained under the same framework. In addition, an experienced radiologist provided a clinical viewpoint on the quality of the learned prototypes, the patch activations, and the global explanations. We achieved a Recall of 0.769 and an area under the receiver operating characteristic curve of 0.719 in our experiments. Even though our findings are non-optimal for entering the clinical practice yet, the radiologist found ProtoPNet’s explanations very intuitive, reporting a high level of satisfaction. Therefore, we believe that prototypical part learning offers a reasonable and promising trade-off between classification performance and the quality of the related explanation.

## Cite our paper

Carloni, G., Berti, A., Iacconi, C., Pascali, M.A., Colantonio, S. (2023). On the Applicability of Prototypical Part Learning in Medical Images: Breast Masses Classification Using ProtoPNet. In: Rousseau, JJ., Kapralos, B. (eds) Pattern Recognition, Computer Vision, and Image Processing. ICPR 2022 International Workshops and Challenges. ICPR 2022. Lecture Notes in Computer Science, vol 13643. Springer, Cham. https://doi.org/10.1007/978-3-031-37660-3_38

### Info and instructions from the originally forked repository

This code package implements the prototypical part network (ProtoPNet) from the paper "This Looks Like That: Deep Learning for Interpretable Image Recognition" (to appear at NeurIPS 2019), by Chaofan Chen* (Duke University), Oscar Li* (Duke University), Chaofan Tao (Duke University), Alina Jade Barnett (Duke University), Jonathan Su (MIT Lincoln Laboratory), and Cynthia Rudin (Duke University) (* denotes equal contribution).
This code package was SOLELY developed by the authors at Duke University, and licensed under MIT License (see LICENSE for more information regarding the use and the distribution of this code package).

Prerequisites: PyTorch, NumPy, cv2, Augmentor (https://github.com/mdbloice/Augmentor). Recommended hardware: 4 NVIDIA Tesla P-100 GPUs or 8 NVIDIA Tesla K-80 GPUs

Instructions for preparing the data:
1. Download the dataset CUB_200_2011.tgz from http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
2. Unpack CUB_200_2011.tgz
3. Crop the images using information from bounding_boxes.txt (included in the dataset)
4. Split the cropped images into training and test sets, using train_test_split.txt (included in the dataset)
5. Put the cropped training images in the directory "./datasets/cub200_cropped/train_cropped/"
6. Put the cropped test images in the directory "./datasets/cub200_cropped/test_cropped/"
7. Augment the training set using img_aug.py (included in this code package)
   -- this will create an augmented training set in the following directory:
      "./datasets/cub200_cropped/train_cropped_augmented/"

Instructions for training the model:
1. In settings.py, provide the appropriate strings for data_path, train_dir, test_dir,
train_push_dir:
(1) data_path is where the dataset resides
    -- if you followed the instructions for preparing the data, data_path should be "./datasets/cub200_cropped/"
(2) train_dir is the directory containing the augmented training set
    -- if you followed the instructions for preparing the data, train_dir should be data_path + "train_cropped_augmented/"
(3) test_dir is the directory containing the test set
    -- if you followed the instructions for preparing the data, test_dir should be data_path + "test_cropped/"
(4) train_push_dir is the directory containing the original (unaugmented) training set
    -- if you followed the instructions for preparing the data, train_push_dir should be data_path + "train_cropped/"
2. Run main.py

Instructions for finding the nearest prototypes to a test image:
1. Run local_analysis.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze
-imgdir is the directory containing the image you want to analyze
-img is the filename of the image you want to analyze
-imgclass is the (0-based) index of the correct class of the image

Instructions for finding the nearest patches to each prototype:
1. Run global_analysis.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to analyze
-model is the filename of the saved model you want to analyze

Instructions for pruning the prototypes from a saved model:
1. Run run_pruning.py and supply the following arguments:
-gpuid is the GPU device ID(s) you want to use (optional, default '0')
-modeldir is the directory containing the model you want to prune prototypes from
-model is the filename of the saved model you want to prune prototypes from
Note: the prototypes in the model must already have been projected (pushed) onto
the nearest latent training patches, before running this script

Instructions for combining several ProtoPNet models (Jupyter Notebook required):
1. Run the Jupyter Notebook combine_models.ipynb
