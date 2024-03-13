import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms


def compose_weak(img, mask, base_size, crop_size):
    img, mask = resize(img, mask, base_size, (0.5, 2.0))
    img, mask = crop(img, mask, crop_size)
    img, mask = hflip(img, mask, 0.5)
    return img, mask

def crop(img, mask, size, mask_redis=None):
    # padding height or width if smaller than cropping size
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    if mask is not None:
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    if mask is not None:
        mask = mask.crop((x, y, x + size, y + size))
    if mask_redis is not None:
        mask_redis = mask_redis.crop((x, y, x + size, y + size))
        return img, mask, mask_redis

    return img, mask


def hflip(img, mask, p, mask_redis=None):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if mask is not None:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if mask_redis is not None:
            mask_redis = mask_redis.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask, mask_redis

    return img, mask


def normalize(img, mask=None, mask_redis=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
    if mask_redis is not None:
        mask_redis = torch.from_numpy(np.array(mask_redis)).long()
        return img, mask, mask_redis

    return img, mask


def resize(img, mask, base_size, ratio_range, mask_redis=None):
    w, h = img.size
    long_side = random.randint(int(base_size * ratio_range[0]), int(base_size * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    if mask is not None:
        mask = mask.resize((ow, oh), Image.NEAREST)
    if mask_redis is not None:
        mask_redis = mask_redis.resize((ow, oh), Image.NEAREST)
        return img, mask, mask_redis

    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def cutout(img, mask, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3,
           ratio_2=1/0.3, value_min=0, value_max=255, pixel_level=True):
    if random.random() < p:
        img = np.array(img)
        mask = np.array(mask)

        img_h, img_w, img_c = img.shape

        while True:
            size = np.random.uniform(size_min, size_max) * img_h * img_w
            ratio = np.random.uniform(ratio_1, ratio_2)
            erase_w = int(np.sqrt(size / ratio))
            erase_h = int(np.sqrt(size * ratio))
            x = np.random.randint(0, img_w)
            y = np.random.randint(0, img_h)

            if x + erase_w <= img_w and y + erase_h <= img_h:
                break

        if pixel_level:
            value = np.random.uniform(value_min, value_max, (erase_h, erase_w, img_c))
        else:
            value = np.random.uniform(value_min, value_max)

        img[y:y + erase_h, x:x + erase_w] = value
        mask[y:y + erase_h, x:x + erase_w] = 255

        img = Image.fromarray(img.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))

    return img, mask
