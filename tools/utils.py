import numpy as np
from PIL import Image
from abc import ABCMeta, abstractmethod
import torch
import collections
import cv2


class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter): pass

class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        else:
            return self.start_lr * (
                    (1 - float(cur_iter) / self.total_iters) ** self.lr_power)

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))  # confusion matrix

    def _fast_hist(self, label_pred, label_true):
        # Follow common experiment, ignore cityscapes background (255), while calculate pascal background (0)
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # count the number of predict classes
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        return iu, np.nanmean(iu)

    def get_class_name(self, dataset_name):
        if dataset_name == 'pascal':
            return ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person',
                    'pottedplant', 'sheep', 'sofa', 'train', 'tv/monitor']
        elif dataset_name == 'cityscapes':
            return ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                    'traffic light', 'traffic sign',
                    'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                    'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        elif dataset_name == 'ucm':  # 0: ignore, 1-17: land cover
            return ['ignore', 'airplane', 'bare soil', 'buildings', 'cars', 'chaparral',
                    'court', 'dock', 'field', 'grass', 'mobile home', 'pavement',
                    'sand', 'sea', 'ship', 'tanks', 'trees', 'water']
        elif dataset_name == 'deepglobe':  # 0-5: land cover, 6: unknown
            return ['urban_land', 'agriculture_land', 'range_land', 'forest_land',
                    'water', 'barren_land']
        elif dataset_name == 'defect_crop' or dataset_name == 'defect_ori':
            return ['background', 'bubble', 'scratch', 'tin_ash']
        elif dataset_name == 'DAGM':
            return ['background', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
        elif dataset_name=='magnetic_tile':
            return ['free(background)','blowhole', 'break', 'crack', 'fray', 'uneven']
        elif dataset_name=='neu_seg':
            return ['background','inclusion', 'patch', 'scratch']


class Evaluator(object):
    def __init__(self, model, num_classes, crop_size, stride_rate):
        self.val_func = model
        self.num_classes = num_classes
        self.crop_size = crop_size
        self.stride_rate = stride_rate

    def scale_process(self, img, ori_shape, crop_size, stride_rate):
        new_rows, new_cols, c = img.shape
        long_size = new_cols if new_cols > new_rows else new_rows

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        if long_size <= min(crop_size[0], crop_size[1]):
            input_data, margin = self.process_image(img, crop_size)  # pad image
            score = self.val_func_process(input_data)
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]
        else:
            stride_0 = int(np.ceil(crop_size[0] * stride_rate))
            stride_1 = int(np.ceil(crop_size[1] * stride_rate))
            img_pad, margin = self.pad_image_to_shape(img, crop_size,
                                                 cv2.BORDER_CONSTANT, value=0)

            pad_rows = img_pad.shape[0]
            pad_cols = img_pad.shape[1]
            r_grid = int(np.ceil((pad_rows - crop_size[0]) / stride_0)) + 1
            c_grid = int(np.ceil((pad_cols - crop_size[1]) / stride_1)) + 1
            data_scale = torch.zeros(self.num_classes, pad_rows, pad_cols).cuda()
            count_scale = torch.zeros(self.num_classes, pad_rows, pad_cols).cuda()

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride_1
                    s_y = grid_yidx * stride_0
                    e_x = min(s_x + crop_size[1], pad_cols)
                    e_y = min(s_y + crop_size[0], pad_rows)
                    s_x = e_x - crop_size[1]
                    s_y = e_y - crop_size[0]
                    img_sub = img_pad[s_y:e_y, s_x: e_x, :]
                    count_scale[:, s_y: e_y, s_x: e_x] += 1

                    input_data, tmargin = self.process_image(img_sub, crop_size)
                    temp_score = self.val_func_process(input_data)
                    temp_score = temp_score[:,
                                 tmargin[0]:(temp_score.shape[1] - tmargin[1]),
                                 tmargin[2]:(temp_score.shape[2] - tmargin[3])]
                    data_scale[:, s_y: e_y, s_x: e_x] += temp_score
            # score = data_scale / count_scale
            score = data_scale
            score = score[:, margin[0]:(score.shape[1] - margin[1]),
                    margin[2]:(score.shape[2] - margin[3])]

        score = score.permute(1, 2, 0)
        data_output = cv2.resize(score.cpu().numpy(),
                                 (ori_shape[1], ori_shape[0]),
                                 interpolation=cv2.INTER_LINEAR)

        return data_output

    def process_image(self, img, crop_size=None):
        p_img = img

        if img.shape[2] < 3:
            im_b = p_img
            im_g = p_img
            im_r = p_img
            p_img = np.concatenate((im_b, im_g, im_r), axis=2)

        if crop_size is not None:
            p_img, margin = self.pad_image_to_shape(p_img, crop_size, cv2.BORDER_CONSTANT, value=0)
            p_img = p_img.transpose(2, 0, 1)

            return p_img, margin

        p_img = p_img.transpose(2, 0, 1)

        return

    def get_2dshape(self, shape, *, zero=True):
        if not isinstance(shape, collections.Iterable):
            shape = int(shape)
            shape = (shape, shape)
        else:
            h, w = map(int, shape)
            shape = (h, w)
        if zero:
            minv = 0
        else:
            minv = 1

        assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
        return shape

    def pad_image_to_shape(self, img, shape, border_mode, value):
        margin = np.zeros(4, np.uint32)
        shape = self.get_2dshape(shape)
        pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
        pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

        margin[0] = pad_height // 2
        margin[1] = pad_height // 2 + pad_height % 2
        margin[2] = pad_width // 2
        margin[3] = pad_width // 2 + pad_width % 2

        img = cv2.copyMakeBorder(img, margin[0], margin[1], margin[2], margin[3],
                                 border_mode, value=value)

        return img, margin

    def normalize(self, img, mean, std):
        # pytorch pretrained model need the input range: 0-1
        img = img.astype(np.float32) / 255.0
        img = img - mean
        img = img / std

        return img

    def val_func_process(self, input_data):
        input_data = np.ascontiguousarray(input_data[None, :, :, :],
                                          dtype=np.float32)
        input_data = torch.FloatTensor(input_data).cuda()  # (1,3,h,w)

        with torch.no_grad():
            score = self.val_func(input_data)  # (1,c,h,w)
            score = score[0]

        return score

    def sliding_eval(self, img):
        _, ori_h, ori_w = img.shape  # (c,h,w)
        img_scale = img.transpose(1, 2, 0)  # (h,w,c)
        # img_scale = cv2.resize(img, None, fx=1, fy=1, interpolation=cv2.INTER_LINEAR)

        # processed_pred = np.zeros((ori_h, ori_w, self.num_classes))
        processed_pred = self.scale_process(img_scale, (ori_h, ori_w),
                                            self.crop_size, self.stride_rate)

        pred = processed_pred.argmax(2)  # (h,w)
        pred = np.expand_dims(pred, axis=0)  # (1,h,w)

        return pred


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    elif dataset == 'ucm':
        cmap[0] = np.array([0, 0, 0])
        cmap[1] = np.array([166, 202, 240])
        cmap[2] = np.array([128, 128, 0])
        cmap[3] = np.array([0, 0, 128])
        cmap[4] = np.array([255, 0, 0])
        cmap[5] = np.array([0, 128, 0])
        cmap[6] = np.array([128, 0, 0])
        cmap[7] = np.array([255, 233, 233])
        cmap[8] = np.array([160, 160, 164])
        cmap[9] = np.array([0, 128, 128])
        cmap[10] = np.array([90, 87, 255])
        cmap[11] = np.array([255, 255, 0])
        cmap[12] = np.array([255, 192, 0])
        cmap[13] = np.array([0, 0, 255])
        cmap[14] = np.array([255, 0, 192])
        cmap[15] = np.array([128, 0, 128])
        cmap[16] = np.array([0, 255, 0])
        cmap[17] = np.array([0, 255, 255])

    elif dataset == 'deepglobe':
        cmap[0] = np.array([0, 255, 255])
        cmap[1] = np.array([255, 255, 0])
        cmap[2] = np.array([255, 0, 255])
        cmap[3] = np.array([0, 255, 0])
        cmap[4] = np.array([0, 0, 255])
        cmap[5] = np.array([255, 255, 255])

    elif dataset == 'defect_crop' or dataset == 'defect_ori':
        cmap[0] = np.array([0, 0, 0])  # background
        cmap[1] = np.array([0, 255, 0])  # bubble
        cmap[2] = np.array([0, 255, 255])  # scratch
        cmap[3] = np.array([255, 255, 0])  # tin_ash

    elif dataset == 'DAGM':
        cmap[0] = np.array([0, 0, 0])  # background
        cmap[1] = np.array([166, 202, 240])
        cmap[2] = np.array([128, 128, 0])
        cmap[3] = np.array([0, 0, 128])
        cmap[4] = np.array([255, 0, 0])
        cmap[5] = np.array([0, 128, 0])
        cmap[6] = np.array([128, 0, 0])
        cmap[7] = np.array([255, 233, 233])
        cmap[8] = np.array([160, 160, 164])
        cmap[9] = np.array([0, 128, 128])
        cmap[10] = np.array([90, 87, 255])

    elif dataset == 'magnetic_tile':
        cmap[0] = np.array([0, 0, 0])  # background
        cmap[1] = np.array([255, 255, 0])  # blowhole
        cmap[2] = np.array([255, 0, 255])  # break
        cmap[3] = np.array([0, 255, 0])  # crack
        cmap[4] = np.array([0, 255, 255])  # fray
        cmap[5] = np.array([255, 0, 0])  # uneven

    elif dataset == 'neu_seg':
        cmap[0] = np.array([0, 0, 0])  # background
        cmap[1] = np.array([166, 202, 240])  # inclusion
        cmap[2] = np.array([128, 128, 0])  # patch
        cmap[3] = np.array([255, 0, 255])  # scratch

    return cmap

def print_iou(iou, mean_pixel_acc=None, class_names=None, show_no_back=False, no_print=False):
    n = iou.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = 'Class %d:' % (i+1)
        else:
            cls = '%d %s' % (i, class_names[i])
        lines.append('%-8s\t%.3f%%' % (cls, iou[i] * 100))
    mean_IoU = np.nanmean(iou)
    # mean_IoU_no_back = np.nanmean(iou[1:])
    # if show_no_back:
    #     lines.append('----------------------------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100, 'mean_IoU_no_back', mean_IoU_no_back*100,
    #                                                                                                 'mean_pixel_ACC',mean_pixel_acc*100))
    # else:
    #     print(mean_pixel_acc)
    #     lines.append('----------------------------     %-8s\t%.3f%%\t%-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100,'mean_pixel_ACC',mean_pixel_acc*100))
    lines.append('----------------------------     %-8s\t%.3f%%' % ('mean_IoU', mean_IoU * 100))
    line = "\n".join(lines)
    if not no_print:
        print(line)
    return line
