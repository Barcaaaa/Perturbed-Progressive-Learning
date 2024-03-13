import cv2

# from dataset.semi import SemiDataset
# from dataset.semi_remote_sensing import SemiDataset
# from dataset.semi_defect import SemiDataset
from dataset.semi_defect import Tile_Dataset_v1

from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3p import DeepLabV3Plus, DeepLabV3Plus_plus_decoder
from model.semseg.pspnet import PSPNet
from tools.utils import count_params, meanIOU, color_map, print_iou, Evaluator

import argparse
import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_defect import label_correct


def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')
    parser.add_argument('--seed', type=int, default=12345)

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['defect_crop', 'DAGM', 'magnetic_tile', 'neu_seg'],
                        default='defect_crop')
    parser.add_argument('--num-classes', type=str, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet50', 'resnet101'], default='resnet18')
    parser.add_argument('--model', type=str, choices=['deeplabv3p', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--best-model-path', type=str, required=True)
    parser.add_argument('--save', dest='save', default=False, action='store_true')
    # parser.add_argument('--unlabeled-id-path', type=str, required=True)
    # parser.add_argument('--reliable-id-path', type=str, required=True)
    # parser.add_argument('--save-pseudo-mask-path', type=str, required=True)
    parser.add_argument('--vis_save', dest='vis_save', default=False, action='store_true')
    parser.add_argument('--vis_path', type=str, required=True)

    parser.add_argument('--stride-rate', type=float, default=2/3)

    args = parser.parse_args()
    return args


def main(args):
    # if not os.path.exists(args.save_pseudo_mask_path):
    #     os.makedirs(args.save_pseudo_mask_path)
    if not os.path.exists(args.vis_path):
        os.makedirs(args.vis_path)

    model_zoo = {'deeplabv3p': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    model = model_zoo[args.model](args.backbone, args.num_classes)

    print('\n================> Evaluation')
    if os.path.splitext(args.best_model_path)[-1] == '.pth':
        model.load_state_dict(torch.load(args.best_model_path))
        model = DataParallel(model).cuda()
        print('\nParams: %.1fM' % count_params(model))
    else:
        print('Invalid model file!')
        return

    model.eval()

    valset = Tile_Dataset_v1(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
    tbar = tqdm(valloader)

    # sliding_evaluator = Evaluator(model, args.num_classes, args.crop_size, args.stride_rate)
    metric = meanIOU(args.num_classes)
    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, idx in tbar:
            img = img.cuda()

            pred = model(img)
            pred = torch.argmax(pred, dim=1)

            metric.add_batch(pred.cpu().numpy(), mask.numpy())

            # for batch_idx in range(img.shape[0]):
            #     pred = sliding_evaluator.sliding_eval(img[batch_idx].cpu().numpy())
            #     metric.add_batch(pred, mask.numpy())

            IOU, mIOU = metric.evaluate()

            if args.vis_save:
                print('\n================> VAL set visualization')

                pred = Image.fromarray(pred.cpu().squeeze(0).numpy().astype(np.uint8), mode='P')
                pred.putpalette(cmap)

                pred.save('%s/%s' % (args.vis_path, os.path.basename(idx[0].split(' ')[1])))

            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

    # print(len(metric.get_class_name()))
    print_iou(IOU, None, metric.get_class_name(args.dataset))


    if args.save:
        print('\n================> Pseudo labelling')
        # valid hard split of one-stage training
        cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
        dataset = Tile_Dataset_v1(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
        # dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                                num_workers=4, drop_last=False)

        pbar = tqdm(dataloader)
        metric = meanIOU(args.num_classes)
        cmap = color_map(args.dataset)

        with torch.no_grad():
            for img, mask, id in pbar:
                img = img.cuda()
                pred = model(img, True)
                pred = torch.argmax(pred, dim=1).cpu()

                metric.add_batch(pred.numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]

                # pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
                # pred.putpalette(cmap)
                #
                # pred.save('%s/%s' % (args.save_pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))

                pbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


if __name__ == '__main__':
    args = parse_args()

    if args.num_classes is None:
        args.num_classes = {'defect_crop': 4, 'DAGM': 11, 'magnetic_tile': 6, 'neu_seg': 4}[args.dataset]
    if args.crop_size is None:
        args.crop_size = {'defect_crop': 512, 'DAGM': 512, 'magnetic_tile': 512, 'neu_seg': 200}[args.dataset]

    print(args)
    main(args)
