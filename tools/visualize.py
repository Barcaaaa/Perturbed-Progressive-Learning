import os
import numpy as np
from PIL import Image


def color_map_tile():
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([0, 0, 0])  # background
    cmap[1] = np.array([255, 255, 0])  # blowhole
    cmap[2] = np.array([255, 0, 255])  # break
    cmap[3] = np.array([0, 255, 0])  # crack
    cmap[4] = np.array([0, 255, 255])  # fray
    cmap[5] = np.array([255, 0, 0])  # uneven
    return cmap

def color_map_neu():
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([0, 0, 0])  # background
    cmap[1] = np.array([166, 202, 240])  # inclusion
    cmap[2] = np.array([128, 128, 0])  # patch
    cmap[3] = np.array([255, 0, 255])  # scratch
    return cmap

# dataset_name = 'magnetic_tile'
dataset_name = 'neu_seg'
if dataset_name == 'magnetic_tile':
    # root_dir = '/data/wuyao/dataset/defect/magnetic_tile/train'
    # root_dir = '/data/wuyao/dataset/defect/magnetic_tile512/train'
    # root_dir = '/data/wuyao/dataset/defect/magnetic_tile/val'
    root_dir = '/data/wuyao/dataset/defect/magnetic_tile512/val'
    mask_dir = os.path.join(root_dir, 'mask')
    vis_dir = os.path.join(root_dir, 'vis_mask')
    cmap = color_map_tile()
elif dataset_name == 'neu_seg':
    root_dir = '/data/xmw/NEU_Seg/annotations'
    mask_dir = os.path.join(root_dir, 'test')
    root_dir2 = '/data/wuyao/dataset/defect/NEU_Seg/annotations'
    vis_dir = os.path.join(root_dir2, 'vis_test')
    cmap = color_map_neu()

if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

for name in os.listdir(mask_dir):
    path = os.path.join(mask_dir, name)
    mask = Image.open(path).convert('L')
    mask.putpalette(cmap)
    mask.save('%s/%s' % (vis_dir, name))
