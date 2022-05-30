#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 17:30:54 2022

@author: si-lab
"""
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('experiment_dir', help='Path to the directory of the experiment containing the 5 folds')
args = parser.parse_args()


experiment_dir = args.experiment_dir
path_to_file = os.path.join(experiment_dir, 'configuration_params.txt')
with open(path_to_file, 'r') as fin:
    for line in fin.readlines(11):
        print(line)