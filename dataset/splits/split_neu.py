import os
import numpy as np
import math
from PIL import Image


numerator = 1
denominator = 2  # 8,4,2
labeled_ratio = numerator / denominator

dataset_type = 'neu_seg'
sample_list = os.path.join(dataset_type, 'train.txt')
sample_list_val = os.path.join(dataset_type, 'val.txt')

# create the output path and save the sublabeled list
out_path = os.path.join(dataset_type, '{}_{}'.format(numerator, denominator))
if not os.path.exists(out_path):
    os.makedirs(out_path)
out_file_labeled = os.path.join(out_path, 'labeled.txt')
out_file_unlabeled = os.path.join(out_path, 'unlabeled.txt')

with open(sample_list, 'r') as f:
    samples = f.read().splitlines()

np.random.shuffle(samples)

# get the subsampled list
labeled_num = int(len(samples) * labeled_ratio)
labeled_list_labeled = samples[:labeled_num]
labeled_list_unlabeled = samples[labeled_num:]

with open(out_file_labeled, 'w') as f:
    for sample in labeled_list_labeled:
        f.write(sample + '\n')

with open(out_file_unlabeled, 'w') as f:
    for sample in labeled_list_unlabeled:
        f.write(sample + '\n')
