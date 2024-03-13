import os
import numpy as np

# set ratio of the labeled samples
numerator = 1
denominator = 8  # 16,8,4,2
labeled_ratio = numerator / denominator

# read the samples list
# dataset_type = 'ucm'
# samples_list = 'F:/dataset/remote_sensing/UC_Merced/UCMerced_LandUse/train.txt'
# samples_list_val = 'F:/dataset/remote_sensing/UC_Merced/UCMerced_LandUse/val.txt'
# dataset_type = 'deepglobe'
# samples_list = 'F:/dataset/remote_sensing/DeepGlobe/train.txt'
# samples_list_val = 'F:/dataset/remote_sensing/DeepGlobe/val.txt'
dataset_type = 'defect_crop'
samples_list = 'F:/dataset/defect_image/phone_screen512x512_ade/train.txt'
samples_list_val = 'F:/dataset/defect_image/phone_screen512x512_ade/val.txt'
# dataset_type = 'defect_ori'
# samples_list = 'F:/dataset/defect_image/dataset/split/train.txt'
# samples_list_val = 'F:/dataset/defect_image/dataset/split/val.txt'

if not os.path.exists(samples_list):
    print('The dataset is not prepared!')
    exit()

with open(samples_list, 'r') as f:
    samples = f.read().splitlines()

np.random.shuffle(samples)

# get the subsampled list
labeled_num = int(len(samples) * labeled_ratio)
labeled_list_labeled = samples[:labeled_num]
labeled_list_unlabeled = samples[labeled_num:]

# create the output path and save the sublabeled list
out_path = os.path.join(dataset_type, '{}_{}'.format(numerator, denominator))
if not os.path.exists(out_path):
    os.makedirs(out_path)
out_file_labeled = os.path.join(out_path, 'labeled.txt')
out_file_unlabeled = os.path.join(out_path, 'unlabeled.txt')

with open(out_file_labeled, 'w') as f:
    for sample in labeled_list_labeled:
        if dataset_type == 'ucm':
            sample_gt = sample.replace('UCMerced_LandUse', 'DLRSD')
            sample_gt = sample_gt.replace('.tif', '.png')
        elif dataset_type == 'deepglobe':
            sample_gt = sample.replace('sat', 'mask')
            sample_gt = sample_gt.replace('.jpg', '.png')
        elif dataset_type == 'defect_crop':
            sample_gt = sample.replace('images', 'annotations')
            sample_gt = sample_gt.replace('.jpg', '.png')
        elif dataset_type == 'defect_ori':
            sample = os.path.join('image', sample)
            sample_gt = sample.replace('image', 'mask')
            sample_gt = sample_gt.replace('.jpg', '.png')
        f.write(sample + ' ' + sample_gt + '\n')

with open(out_file_unlabeled, 'w') as f:
    for sample in labeled_list_unlabeled:
        if dataset_type == 'ucm':
            sample_gt = sample.replace('UCMerced_LandUse', 'DLRSD')
            sample_gt = sample_gt.replace('.tif', '.png')
        elif dataset_type == 'deepglobe':
            sample_gt = sample.replace('sat', 'mask')
            sample_gt = sample_gt.replace('.jpg', '.png')
        elif dataset_type == 'defect_crop':
            sample_gt = sample.replace('images', 'annotations')
            sample_gt = sample_gt.replace('.jpg', '.png')
        elif dataset_type == 'defect_ori':
            sample = os.path.join('image', sample)
            sample_gt = sample.replace('image', 'mask')
            sample_gt = sample_gt.replace('.jpg', '.png')
        f.write(sample + ' ' + sample_gt + '\n')

# if not os.path.exists(samples_list_val):
#     print('The dataset is not prepared!')
#     exit()
#
# with open(samples_list_val, 'r') as f:
#     samples = f.read().splitlines()
# out_file_val = os.path.join(dataset_type, 'val.txt')
#
# with open(out_file_val, 'w') as f:
#     for sample in samples:
#         if dataset_type == 'ucm':
#             sample_gt = sample.replace('UCMerced_LandUse', 'DLRSD')
#             sample_gt = sample_gt.replace('.tif', '.png')
#         elif dataset_type == 'deepglobe':
#             sample_gt = sample.replace('sat', 'mask')
#             sample_gt = sample_gt.replace('.jpg', '.png')
#         elif dataset_type == 'defect_crop':
#             sample_gt = sample.replace('images', 'annotations')
#             sample_gt = sample_gt.replace('.jpg', '.png')
#         elif dataset_type == 'defect_ori':
#             sample = os.path.join('image', sample)
#             sample_gt = sample.replace('image', 'mask')
#             sample_gt = sample_gt.replace('.jpg', '.png')
#         f.write(sample + ' ' + sample_gt + '\n')

