import os
import numpy as np
import math
from PIL import Image


numerator = 1
denominator = 2  # 8,4,2
labeled_ratio = numerator / denominator

root_dir = '/data/wuyao/dataset/defect/magnetic_tile/train'
mask_dir = os.path.join(root_dir, 'mask')
cls_num = np.zeros(6, dtype='int')
cls_list = [[] for _ in range(6)]
for name in os.listdir(mask_dir):
    path = os.path.join(mask_dir, name)
    mask = Image.open(path).convert('L')
    cls_idx = np.unique(np.array(mask))
    if len(cls_idx) == 1 and cls_idx[0] == 0:
        cls_num[0] += 1
        cls_list[0].append('train/mask/' + name)
    for i in cls_idx:
        if i != 0:
            cls_num[i] += 1
            cls_list[i].append('train/mask/' + name)

# print(cls_num)  # [764 92 68 45 25 79]

dataset_type = 'magnetic_tile'
# create the output path and save the sublabeled list
out_path = os.path.join(dataset_type, '{}_{}'.format(numerator, denominator))
if not os.path.exists(out_path):
    os.makedirs(out_path)
out_file_labeled = os.path.join(out_path, 'labeled.txt')
out_file_unlabeled = os.path.join(out_path, 'unlabeled.txt')

labeled_list_labeled = [[] for _ in range(6)]
labeled_list_unlabeled = [[] for _ in range(6)]
for j in range(6):
    samples = cls_list[j]
    np.random.shuffle(samples)
    labeled_num = math.ceil(len(cls_list[j]) * labeled_ratio)
    print(labeled_num)
    labeled_list_labeled[j] = samples[:labeled_num]
    labeled_list_unlabeled[j] = samples[labeled_num:]

with open(out_file_labeled, 'w') as f:
    for j in range(6):
        for sample_gt in labeled_list_labeled[j]:
            if dataset_type == 'magnetic_tile':
                sample = sample_gt.replace('mask', 'image')
                sample = sample.replace('.png', '.jpg')
            f.write(sample + ' ' + sample_gt + '\n')

with open(out_file_unlabeled, 'w') as f:
    for j in range(6):
        for sample_gt in labeled_list_unlabeled[j]:
            if dataset_type == 'magnetic_tile':
                sample = sample_gt.replace('mask', 'image')
                sample = sample.replace('.png', '.jpg')
            f.write(sample + ' ' + sample_gt + '\n')
