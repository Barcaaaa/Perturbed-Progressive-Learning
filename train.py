import os
import time
import argparse
from copy import deepcopy
import numpy as np
from PIL import Image
import random
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.semi_defect import Tile_Dataset_v1
from dataset.data import TwoStreamBatchSampler
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3p import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from model.semseg.fpn import FCN
from tools.utils import count_params, meanIOU, color_map, print_iou, Evaluator
from tools.funcs import BoxMaskGenerator

MODE = None

def parse_args():
    parser = argparse.ArgumentParser(description='Framework')
    parser.add_argument('--seed', type=int, default=12345)

    # basic settings
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['defect_crop', 'magnetic_tile', 'neu_seg'],
                        default='defect_crop')
    parser.add_argument('--num-classes', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet18', 'resnet50', 'resnet50_v1'],
                        default='resnet18')
    parser.add_argument('--model', type=str, choices=['deeplabv3p', 'pspnet', 'deeplabv2', 'fpn'],
                        default='deeplabv3p')
    parser.add_argument('--sliding_eval', type=bool, default=None)
    parser.add_argument('--stride-rate', type=float, default=2/3)

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--use_PPL', default=False, action='store_true')

    parser.add_argument('--use_cutmix', type=bool, default=False)
    parser.add_argument('--cutmix_mask_prop_range', type=list, default=[0.25, 0.5])  # crop size of mask
    parser.add_argument('--cutmix_boxmask_n_boxes', type=int, default=3)  # num of mask
    parser.add_argument('--cutmix_boxmask_fixed_aspect_ratio', type=bool, default=True)
    parser.add_argument('--cutmix_boxmask_by_size', type=bool, default=True)
    parser.add_argument('--cutmix_boxmask_outside_bounds', type=bool, default=True)
    parser.add_argument('--cutmix_boxmask_no_invert', type=bool, default=True)

    args = parser.parse_args()
    return args

def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.use_PPL and args.reliable_id_path is None:
        exit('Please specify reliable-id-path.')

    criterion = CrossEntropyLoss(ignore_index=255)
    valset = Tile_Dataset_v1(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=4, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    global MODE

    print('\n\n\n================> Total stage 1/6: Supervised training with labeled images')
    model_zoo = {'deeplabv3p': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2, 'fpn': FCN}
    model = model_zoo[args.model](args.backbone, args.num_classes)

    head_lr_multiple = 10.0
    if args.model == 'fpn':
        optimizer = SGD([{'params': model.parameters(), 'lr': args.lr}],
                        lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                         {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                          'lr': args.lr * head_lr_multiple}],
                        lr=args.lr, momentum=0.9, weight_decay=1e-4)

    MODE = 'train'
    trainset = Tile_Dataset_v1(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    # trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                             num_workers=4, drop_last=True)
    model = DataParallel(model).cuda()
    print('\nParams: %.1fM' % count_params(model))

    best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)

    if args.use_PPL:
        print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')
        dataset = Tile_Dataset_v1(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                                num_workers=args.n_workers, drop_last=False)
        select_reliable_unreliable(checkpoints, dataloader, args)

        print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')
        cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
        dataset = Tile_Dataset_v1(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                                num_workers=args.n_workers, drop_last=False)
        pseudo_labeling(best_model, dataloader, args)

        print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')
        MODE = 'semi_train'
        args.use_cutmix = True
        trainset = Tile_Dataset_v1(args.dataset, args.data_root, MODE, args.crop_size,
                                   args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path)
        labeled_list = trainset.labeled_ids
        unlabeled_list = trainset.unlabeled_ids
        # Custom labeled and unlabeled batch size ratio
        train_sampler = TwoStreamBatchSampler(labeled_list, unlabeled_list, args.batch_size // 2, args.batch_size // 2)
        trainloader = DataLoader(trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=args.n_workers)
        model, optimizer = init_basic_elems(args)

        best_model = train(model, trainloader, valloader, criterion, optimizer, args)
        # model.load_state_dict(torch.load('checkpoint.pth'))
        # best_model = model.cuda()
        # print("Load Model checkpoint!")

        print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')
        cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
        dataset = Tile_Dataset_v1(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                                num_workers=args.n_workers, drop_last=False)
        pseudo_labeling(best_model, dataloader, args)

        # <================================== The 2nd stage re-training ==================================>
        print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')

        MODE = 'semi_train'
        trainset = Tile_Dataset_v1(args.dataset, args.data_root, MODE, args.crop_size,
                                   args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
        labeled_list = trainset.labeled_ids
        unlabeled_list = trainset.unlabeled_ids
        train_sampler = TwoStreamBatchSampler(labeled_list, unlabeled_list, args.batch_size // 2, args.batch_size // 2)
        trainloader = DataLoader(trainset, batch_sampler=train_sampler, pin_memory=True, num_workers=args.n_workers)
        model, optimizer = init_basic_elems(args)

        train(model, trainloader, valloader, criterion, optimizer, args)

    return

def init_basic_elems(args):
    model_zoo = {'deeplabv3p': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2, 'fpn': FCN}
    model = model_zoo[args.model](args.backbone, args.num_classes)

    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        # model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    if args.model == 'fpn':
        optimizer = SGD([{'params': model.parameters(), 'lr': args.lr}],
                        lr=args.lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                         {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                          'lr': args.lr * head_lr_multiple}],
                        lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()
    return model, optimizer

def train(model, trainloader, valloader, criterion, optimizer, args):
    iters = 0
    total_iters = len(trainloader) * args.epochs
    previous_best = 0.0

    if args.use_cutmix:
        mask_generator = BoxMaskGenerator(
            prop_range=args.cutmix_mask_prop_range,
            n_boxes=args.cutmix_boxmask_n_boxes,
            random_aspect_ratio=args.cutmix_boxmask_fixed_aspect_ratio,
            prop_by_area=args.cutmix_boxmask_by_size,
            within_bounds=args.cutmix_boxmask_outside_bounds,
            invert=args.cutmix_boxmask_no_invert
        )

    global MODE
    if MODE == 'train':
        checkpoints = []

    for epoch in range(args.epochs):
        total_loss = 0.0
        print("\n==> Epoch %i, learning rate = %f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        ''' training '''
        model.train()
        tbar = tqdm(trainloader)
        for i, (img, mask, idx) in enumerate(tbar):
            img, mask = img.cuda(), mask.cuda()
            if not args.use_cutmix:
                pred = model(img)
                loss = criterion(pred, mask)
            else:
                # ============================== Label-Efficient CutMix ==============================
                b, c, h, w = img.shape
                # NOTE: Do not apply strong augmentation to unlabeled images when calculating confidence scores
                with torch.no_grad():
                    model.eval()
                    pred_u = model(img[b // 2:])  # unlabeled prediction
                    pred_u = F.softmax(pred_u, dim=1)
                    logits_u, _ = torch.max(pred_u, dim=1)  # (b/2,h,w)
                    entropy = -torch.sum(pred_u * torch.log(pred_u + 1e-10), dim=1)  # (b/2,h,w)
                    entropy /= np.log(args.num_classes)
                    confidence = 1.0 - entropy
                    confidence = confidence * logits_u  # confidence * prob
                    confidence = confidence.mean(dim=[1, 2])  # (b/2,1) final decision score
                    confidence = confidence.cpu().numpy().tolist()
                    del pred_u

                model.train()
                img_l, mask_l = img[:b // 2], mask[:b // 2]
                img_u_ori, mask_u = img[b // 2:], mask[b // 2:]

                # Strong Augmentation
                from torchvision import transforms
                from dataset.transform import blur

                trans_pil = transforms.ToPILImage()
                img_u_aug = []
                for batch_idx in range(b // 2):
                    img_u = trans_pil(img_u_ori[batch_idx])
                    if random.random() < 0.8:
                        img_u = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_u)
                    img_u = transforms.RandomGrayscale(p=0.2)(img_u)
                    img_u = blur(img_u, p=0.5)
                    img_u = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])(img_u)
                    img_u_aug.append(img_u.unsqueeze(0))
                img_u_aug = torch.cat(img_u_aug, dim=0).cuda()

                # Step 1. labeled and unlabeled cutmix
                batch_mix_masks_1 = mask_generator.generate_params(1, (h, w))
                batch_mix_masks_1 = torch.from_numpy(batch_mix_masks_1.astype(np.float32)).cuda()

                img_u_list_1, mask_u_list_1 = [], []
                for batch_idx in range(b // 2):
                    if np.random.random() > confidence[batch_idx]:
                        img_mix = img_u_aug[batch_idx] * (1 - batch_mix_masks_1) + img_l[batch_idx] * batch_mix_masks_1
                        mask_mix = mask_u[batch_idx] * (1 - batch_mix_masks_1) + mask_l[batch_idx] * batch_mix_masks_1
                        mask_mix = mask_mix.squeeze(1).long()
                    else:
                        img_mix = img_u_aug[batch_idx].unsqueeze(0)
                        mask_mix = mask_u[batch_idx].unsqueeze(0)
                    img_u_list_1.append(img_mix)
                    mask_u_list_1.append(mask_mix)

                img_u_mix_1 = torch.cat(img_u_list_1, dim=0)
                mask_u_mix_1 = torch.cat(mask_u_list_1, dim=0)

                # Step 2. two different unlabeled cutmix
                batch_mix_masks_2 = mask_generator.generate_params(1, (h, w))
                batch_mix_masks_2 = torch.from_numpy(batch_mix_masks_2.astype(np.float32)).cuda()

                img_u_list_2, mask_u_list_2 = [], []
                for batch_idx in range(b // 2):
                    img_mix = img_u_aug[batch_idx] * (1 - batch_mix_masks_2) + \
                              img_u_mix_1[(batch_idx + 1) % (b // 2)] * batch_mix_masks_2
                    mask_mix = mask_u[batch_idx] * (1 - batch_mix_masks_2) + \
                               mask_u_mix_1[(batch_idx + 1) % (b // 2)] * batch_mix_masks_2
                    mask_mix = mask_mix.squeeze(1).long()
                    img_u_list_2.append(img_mix)
                    mask_u_list_2.append(mask_mix)

                img_u_mix_2 = torch.cat(img_u_list_2, dim=0)
                mask_u_mix_2 = torch.cat(mask_u_list_2, dim=0)

                # new batch
                img_new = torch.cat([img_l, img_u_mix_2], dim=0)
                mask_new = torch.cat([mask_l, mask_u_mix_2], dim=0)

                pred = model(img_new)
                loss = criterion(pred, mask_new)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            if args.model == 'fpn':
                optimizer.param_groups[0]["lr"] = lr
            else:
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

            tbar.set_description('Loss: %.4f' % (total_loss / (i + 1)))

        ''' validation '''
        model.eval()
        tbar = tqdm(valloader)
        metric = meanIOU(args.num_classes)

        if args.sliding_eval:
            sliding_evaluator = Evaluator(model, args.num_classes, args.crop_size, args.stride_rate)

        with torch.no_grad():
            for img, mask, id in tbar:
                img = img.cuda()

                if args.sliding_eval:
                    for batch_idx in range(img.shape[0]):
                        pred = sliding_evaluator.sliding_eval(img[batch_idx].cpu().numpy())
                        metric.add_batch(pred, mask.numpy())
                else:
                    pred = model(img)
                    pred = torch.argmax(pred, dim=1)
                    metric.add_batch(pred.cpu().numpy(), mask.numpy())

                IOU, mIOU = metric.evaluate()
                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

        with open(args.save_path + "/log_PPL.txt",encoding="utf-8",mode="a+") as f:
            f.write('\n========================\n')
            f.write(f'epoch = {epoch}/{args.epochs}, previous best mIoU = {previous_best}\n')
            f.write(print_iou(IOU, None, metric.get_class_name(args.dataset)))
            f.write('\n')

        mIOU *= 100.0
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))
            best_model = deepcopy(model)
        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model

def select_reliable_unreliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)
    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []
    with torch.no_grad():
        for img, _, id in tbar:
            img = img.cuda()
            preds = []
            for model in models:
                pred_tmp = model(img)
                preds.append(torch.argmax(pred_tmp, dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                metric = meanIOU(args.num_classes)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])
            confidence_1 = sum(mIOU) / len(mIOU)

            pred_u = F.softmax(pred_tmp, dim=1)
            logits_u, _ = torch.max(pred_u, dim=1)
            entropy = -torch.sum(pred_u * torch.log(pred_u + 1e-10), dim=1)
            entropy /= np.log(args.num_classes)
            confidence_2 = 1.0 - entropy
            confidence_2 = confidence_2 * logits_u
            confidence_2 = confidence_2.mean(dim=[1, 2])
            confidence_2 = confidence_2.cpu().numpy().tolist()

            reliability = confidence_1 * confidence_2[0]
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:len(id_to_reliability) // 2]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[len(id_to_reliability) // 2:]:
            f.write(elem[0] + '\n')

def pseudo_labeling(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)
    metric = meanIOU(args.num_classes)
    cmap = color_map(args.dataset)
    if args.sliding_eval:
        sliding_evaluator = Evaluator(model, args.num_classes, args.crop_size, args.stride_rate)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            if args.sliding_eval:
                for batch_idx in range(img.shape[0]):
                    pred = sliding_evaluator.sliding_eval(img[batch_idx].cpu().numpy())
                    metric.add_batch(pred, mask.numpy())
            else:
                pred = model(img, True)
                pred = torch.argmax(pred, dim=1).cpu()
                metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)
            pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))
            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


if __name__ == '__main__':
    args = parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    if args.epochs is None:  # split-- magnetic_tile: 115/134/200   neu_seg: 54/58/67/100
        args.epochs = {'defect_crop': 60, 'magnetic_tile': 200, 'neu_seg': 100}[args.dataset]
    if args.batch_size is None:
        args.batch_size = 8
    if args.num_classes is None:
        args.num_classes = {'defect_crop': 4, 'magnetic_tile': 6, 'neu_seg': 4}[args.dataset]
    if args.lr is None:
        args.lr = {'defect_crop': 0.001, 'magnetic_tile': 0.001, 'neu_seg': 0.01}[args.dataset] / 8 * args.batch_size
    if args.crop_size is None:
        args.crop_size = {'defect_crop': 512, 'magnetic_tile': 512, 'neu_seg': 200}[args.dataset]
    if args.sliding_eval is None:
        args.sliding_eval = False

    print(args)
    main(args)
