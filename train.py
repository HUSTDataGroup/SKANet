import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from trainer import trainer_acdc, trainer_mscmrseg

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='ACDC', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_ACDC', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=400, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=4, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=1e-4,
                    help='segmentation network learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='optimizer weight decay')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    dataset_name = args.dataset
    # ACDC, MSCMRSeg
    dataset_config = {
        'ACDC': {
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': 4,
        },
        'MSCMRSeg': {
            'root_path': '../data/MSCMRSeg',
            'list_dir': './lists/lists_MSCMRSeg',
            'num_classes': 4,
        },
    }
    
    if dataset_name in dataset_config:
        args.num_classes = dataset_config[dataset_name]['num_classes']
        args.root_path = dataset_config[dataset_name]['root_path']
        args.list_dir = dataset_config[dataset_name]['list_dir']

    args.is_pretrain = True
    
    # SKANet
    args.exp = 'SKANet_' + dataset_name + '_' + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'SKANet')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_s' + str(args.seed)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
        
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    if args.is_pretrain and os.path.exists(config_vit.pretrained_path):
        net.load_from(weights=np.load(config_vit.pretrained_path))
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.base_lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    trainer = {'ACDC': trainer_acdc,'MSCMRSeg': trainer_mscmrseg,}
    trainer[dataset_name](args, net, snapshot_path, optimizer, scheduler)