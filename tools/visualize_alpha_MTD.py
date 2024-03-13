import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


raw_image_dir = '/data/wuyao/dataset/defect/magnetic_tile512/val/image'

### GT blend
raw_mask_dir = '/data/wuyao/dataset/defect/magnetic_tile512/val/vis_mask'
raw_blend_dir = '/data/wuyao/dataset/defect/magnetic_tile512/val/vis_mask_blend'
if not os.path.exists(raw_blend_dir):
    os.makedirs(raw_blend_dir)

# 将图像和标签 透明化混合
for idx, name in enumerate(os.listdir(raw_image_dir)):
    img_path = os.path.join(raw_image_dir, name)
    img = Image.open(img_path).convert('RGBA')
    name = os.path.splitext(name)[0] + '.png'
    mask_path = os.path.join(raw_mask_dir, name)
    mask = Image.open(mask_path).convert('RGBA')
    x, y = mask.size
    for i in range(x):
        for j in range(y):
            color = mask.getpixel((i, j))
            value = np.mean(list(color[:-1]))
            if value == 0:
                color = color[:-1] + (1,)  # 背景完全透明化
            else:
                color = color[:-1] + (50,)  # 前景缺陷半透明
            mask.putpixel((i, j), color)

    # PS:之前用PIL的paste将两张图片重叠，后者尽管透明度为0，也会完全覆盖前者，不适用
    plt.axis('off')  # 去掉轴坐标
    plt.imshow(img)
    plt.imshow(mask)
    plt.savefig('%s/%s' % (raw_blend_dir, name), bbox_inches='tight',pad_inches=0)  # 去掉白边

    plt.clf()  # 保存完，需清除图形，否则会内存泄漏
    print(idx, name)

### PPL blend
pred_mask_dir = '/data/wuyao/code/PPL/outdir/vis_result/magnetic_tile/1_8_r18_sa_amix'
pred_blend_dir = '/data/wuyao/code/PPL/outdir/vis_result/magnetic_tile/1_8_r18_sa_amix_blend'
if not os.path.exists(pred_blend_dir):
    os.makedirs(pred_blend_dir)

# 将图像和预测 透明化混合
for idx, name in enumerate(os.listdir(raw_image_dir)):
    img_path = os.path.join(raw_image_dir, name)
    img = Image.open(img_path).convert('RGBA')
    name = os.path.splitext(name)[0] + '.png'
    mask_path = os.path.join(pred_mask_dir, name)
    mask = Image.open(mask_path).convert('RGBA')
    x, y = mask.size
    for i in range(x):
        for j in range(y):
            color = mask.getpixel((i, j))
            value = np.mean(list(color[:-1]))
            if value == 0:
                color = color[:-1] + (1,)  # 背景完全透明化
            else:
                color = color[:-1] + (50,)  # 前景缺陷半透明
            mask.putpixel((i, j), color)

    # PS:之前用PIL的paste将两张图片重叠，后者尽管透明度为0，也会完全覆盖前者，不适用
    plt.axis('off')  # 去掉轴坐标
    plt.imshow(img)
    plt.imshow(mask)
    plt.savefig('%s/%s' % (pred_blend_dir, name), bbox_inches='tight',pad_inches=0)  # 去掉白边

    plt.clf()
    print(idx, name)

### CCT blend
pred_mask_dir = '/data/wuyao/code/PPL/outdir/vis_result/magnetic_tile/CCT'
pred_blend_dir = '/data/wuyao/code/PPL/outdir/vis_result/magnetic_tile/CCT_blend'
if not os.path.exists(pred_blend_dir):
    os.makedirs(pred_blend_dir)

# 将图像和预测 透明化混合
for idx, name in enumerate(os.listdir(raw_image_dir)):
    img_path = os.path.join(raw_image_dir, name)
    img = Image.open(img_path).convert('RGBA')
    name = os.path.splitext(name)[0] + '.png'
    mask_path = os.path.join(pred_mask_dir, name)
    mask = Image.open(mask_path).convert('RGBA')
    x, y = mask.size
    for i in range(x):
        for j in range(y):
            color = mask.getpixel((i, j))
            value = np.mean(list(color[:-1]))
            if value == 0:
                color = color[:-1] + (1,)  # 背景完全透明化
            else:
                color = color[:-1] + (50,)  # 前景缺陷半透明
            mask.putpixel((i, j), color)

    # PS:之前用PIL的paste将两张图片重叠，后者尽管透明度为0，也会完全覆盖前者，不适用
    plt.axis('off')  # 去掉轴坐标
    plt.imshow(img)
    plt.imshow(mask)
    plt.savefig('%s/%s' % (pred_blend_dir, name), bbox_inches='tight',pad_inches=0)  # 去掉白边

    plt.clf()
    print(idx, name)

### CPS blend
pred_mask_dir = '/data/wuyao/code/PPL/outdir/vis_result/magnetic_tile/CPS'
pred_blend_dir = '/data/wuyao/code/PPL/outdir/vis_result/magnetic_tile/CPS_blend'
if not os.path.exists(pred_blend_dir):
    os.makedirs(pred_blend_dir)

# 将图像和预测 透明化混合
for idx, name in enumerate(os.listdir(raw_image_dir)):
    img_path = os.path.join(raw_image_dir, name)
    img = Image.open(img_path).convert('RGBA')
    name = os.path.splitext(name)[0] + '.png'
    mask_path = os.path.join(pred_mask_dir, name)
    mask = Image.open(mask_path).convert('RGBA')
    x, y = mask.size
    for i in range(x):
        for j in range(y):
            color = mask.getpixel((i, j))
            value = np.mean(list(color[:-1]))
            if value == 0:
                color = color[:-1] + (1,)  # 背景完全透明化
            else:
                color = color[:-1] + (50,)  # 前景缺陷半透明
            mask.putpixel((i, j), color)

    # PS:之前用PIL的paste将两张图片重叠，后者尽管透明度为0，也会完全覆盖前者，不适用
    plt.axis('off')  # 去掉轴坐标
    plt.imshow(img)
    plt.imshow(mask)
    plt.savefig('%s/%s' % (pred_blend_dir, name), bbox_inches='tight',pad_inches=0)  # 去掉白边

    plt.clf()
    print(idx, name)

### MT blend
pred_mask_dir = '/data/wuyao/code/PPL/outdir/vis_result/magnetic_tile/MT'
pred_blend_dir = '/data/wuyao/code/PPL/outdir/vis_result/magnetic_tile/MT_blend'
if not os.path.exists(pred_blend_dir):
    os.makedirs(pred_blend_dir)

# 将图像和预测 透明化混合
for idx, name in enumerate(os.listdir(raw_image_dir)):
    img_path = os.path.join(raw_image_dir, name)
    img = Image.open(img_path).convert('RGBA')
    name = os.path.splitext(name)[0] + '.png'
    mask_path = os.path.join(pred_mask_dir, name)
    mask = Image.open(mask_path).convert('RGBA')
    x, y = mask.size
    for i in range(x):
        for j in range(y):
            color = mask.getpixel((i, j))
            value = np.mean(list(color[:-1]))
            if value == 0:
                color = color[:-1] + (1,)  # 背景完全透明化
            else:
                color = color[:-1] + (50,)  # 前景缺陷半透明
            mask.putpixel((i, j), color)

    # PS:之前用PIL的paste将两张图片重叠，后者尽管透明度为0，也会完全覆盖前者，不适用
    plt.axis('off')  # 去掉轴坐标
    plt.imshow(img)
    plt.imshow(mask)
    plt.savefig('%s/%s' % (pred_blend_dir, name), bbox_inches='tight',pad_inches=0)  # 去掉白边

    plt.clf()
    print(idx, name)
