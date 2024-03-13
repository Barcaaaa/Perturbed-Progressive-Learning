import os
import numpy as np


# set ratio of the labeled samples
numerator = 1
denominator = 16  # 16,8,4,2
cps_dataset_type = 'DAGM_cps'
dataset_type = 'DAGM'

out_path = os.path.join(dataset_type, '{}_{}'.format(numerator, denominator))
if not os.path.exists(out_path):
    os.makedirs(out_path)

out_file_labeled = os.path.join(out_path, 'labeled.txt')
out_file_unlabeled = os.path.join(out_path, 'unlabeled.txt')
out_file_val = os.path.join(dataset_type, 'val.txt')

subsample_file = os.path.join(cps_dataset_type, '{}_{}'.format(numerator, denominator))
subsample_labeled = os.path.join(subsample_file, 'labeled.txt')
subsample_unlabeled = os.path.join(subsample_file, 'unlabeled.txt')

with open(subsample_labeled, 'r') as f:
    labeled_list_labeled = f.read().splitlines()

with open(subsample_unlabeled, 'r') as f:
    labeled_list_unlabeled = f.read().splitlines()

with open(out_file_labeled, 'w') as f:
    for sample in labeled_list_labeled:
        sample = 'train/image/' + sample
        sample_gt = sample.replace('image', 'mask')
        f.write(sample + ' ' + sample_gt + '\n')

with open(out_file_unlabeled, 'w') as f:
    for sample in labeled_list_unlabeled:
        sample = 'train/image/' + sample
        sample_gt = sample.replace('image', 'mask')
        f.write(sample + ' ' + sample_gt + '\n')

subsample_val = os.path.join(cps_dataset_type, 'val.txt')

with open(subsample_val, 'r') as f:
    val_list = f.read().splitlines()

with open(out_file_val, 'w') as f:
    for sample in val_list:
        sample = 'val/image/' + sample
        sample_gt = sample.replace('image', 'mask')
        f.write(sample + ' ' + sample_gt + '\n')

