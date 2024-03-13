import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from dataset.transform import crop, hflip, normalize, resize, blur, cutout


def fft(img):
    # gamma变换
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.copy(img / 255.0)
    img = 5 * np.power(img, 0.9)
    img[img > 1] = 1

    # 傅里叶变换
    dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
    # 获得频谱图，将低频值转换到中间
    dft_shift = np.fft.fftshift(dft)
    # 获取频率为0部分中心点位置
    rows, cols = img.shape  # (471,498),分别保存图像的高和宽
    crow, col = int(rows / 2), int(cols / 2)  # 计算中心点坐标
    # 构造低通滤波器，相当于构造一个掩模
    mask = np.zeros((rows, cols, 2), np.uint8)  # 构造的size和原图相同，2通道，傅里叶变换后有实部和虚部
    mask[crow - 50:crow + 50, col - 50:col + 50] = 255  # 构造一个以频率为0点中心对称，长30+30，宽30+30的一个区域，只保留区域内部的频率
    # 频谱图上，低频的信息都在中间，滤波器和频谱图相乘，遮挡四周，保留中间，中间是低频
    fshift = dft_shift * mask
    # 在获得频谱图时，将低频点从边缘点移动到图像中间，现在要逆变换，得还回去
    f_ishift = np.fft.ifftshift(fshift)
    # 傅里叶逆变换
    img_back = cv2.idft(f_ishift)
    # 还原后的还是有实部和虚部，需要进一步处理
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    temp = np.zeros_like(img_back)
    cv2.normalize(img_back, dst=temp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    temp = np.uint8(temp)
    temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB)
    temp = Image.fromarray(temp)

    return temp


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None,
                 pseudo_mask_path=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
        else:
            if mode == 'val':
                id_path = 'dataset/splits/%s/val.txt' % name
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        if self.mode == 'semi_train':
            id = item
        else:
            id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0]))
        if len(img.split()) == 1:  # DAGM是单通道图像，复制到3通道
            img = img.convert('RGB')

        # img = fft(img)
        if self.mode == 'val' or self.mode == 'label':
            mask = Image.open(os.path.join(self.root, id.split(' ')[1])).convert('L')
            img, mask = normalize(img, mask)
            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(' ')[1]))
        else:
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # weak augmentation on all training images
        base_size = 512
        img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, 0.5)

        # strong augmentation on unlabeled images (unavailable for mobile screen)
        # if self.mode == 'semi_train' and id in self.unlabeled_ids:
        #     if random.random() < 0.8:
        #         img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
        #     img = transforms.RandomGrayscale(p=0.2)(img)
        #     img = blur(img, p=0.5)
        #     # img, mask = cutout(img, mask, p=0.5)  # comment if use cutmix

        img, mask = normalize(img, mask)

        return img, mask, id

    def __len__(self):
        return len(self.ids)


class Tile_Dataset(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None,
                 pseudo_mask_path=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
        else:
            if mode == 'val':
                id_path = 'dataset/splits/magnetic_tile/val.txt'
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        base_size = 512

        if self.mode == 'semi_train':
            id = item
        else:
            id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')

        # img = fft(img)
        if self.mode == 'val' or self.mode == 'label':
            mask = Image.open(os.path.join(self.root, id.split(' ')[1])).convert('L')
            img = img.resize((base_size, base_size), Image.BILINEAR)
            mask = mask.resize((base_size, base_size), Image.NEAREST)
            img, mask = normalize(img, mask)

            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(' ')[1])).convert('L')
        else:
            # mode == 'semi_train' and the id corresponds to unlabeled image
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # weak augmentation on all training images
        # img, mask = resize(img, mask, base_size, (0.5, 2.0))
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size,base_size), Image.NEAREST)

        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, 0.5)

        # strong augmentation on unlabeled images
        if self.mode == 'semi_train' and id in self.unlabeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)

        img, mask = normalize(img, mask)

        return img, mask, id

    def __len__(self):
        return len(self.ids)


class Tile_Dataset_v1(Dataset):
    def __init__(self, name, root, mode, size, labeled_id_path=None, unlabeled_id_path=None,
                 pseudo_mask_path=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.pseudo_mask_path = pseudo_mask_path

        if mode == 'semi_train':
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
        else:
            if mode == 'val':
                if name == 'magnetic_tile':
                    id_path = 'dataset/splits/magnetic_tile/val.txt'
                elif name == 'neu_seg':
                    id_path = 'dataset/splits/neu_seg/val.txt'
                elif name == 'defect_crop':
                    id_path = 'dataset/splits/defect_crop/val.txt'
            elif mode == 'label':
                id_path = unlabeled_id_path
            elif mode == 'train':
                id_path = labeled_id_path

            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        if self.name == 'magnetic_tile' or self.name == 'defect_crop':
            base_size = 512
        elif self.name == 'neu_seg':
            base_size = 200

        if self.mode == 'semi_train':
            id = item
        else:
            id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')

        # img = fft(img)
        if self.mode == 'val' or self.mode == 'label':
            mask = Image.open(os.path.join(self.root, id.split(' ')[1])).convert('L')

            img = img.resize((base_size, base_size), Image.BILINEAR)
            mask = mask.resize((base_size, base_size), Image.NEAREST)
            img, mask = normalize(img, mask)

            return img, mask, id

        if self.mode == 'train' or (self.mode == 'semi_train' and id in self.labeled_ids):
            mask = Image.open(os.path.join(self.root, id.split(' ')[1])).convert('L')
        else:
            fname = os.path.basename(id.split(' ')[1])
            mask = Image.open(os.path.join(self.pseudo_mask_path, fname))

        # weak augmentation on all training images
        if self.name == 'defect_crop':
            img, mask = resize(img, mask, base_size, (0.5, 2.0))
        else:
            img = img.resize((base_size, base_size), Image.BILINEAR)
            mask = mask.resize((base_size, base_size), Image.NEAREST)

        img, mask = crop(img, mask, self.size)
        img, mask = hflip(img, mask, 0.5)

        img, mask = normalize(img, mask)

        return img, mask, id

    def __len__(self):
        return len(self.ids)
