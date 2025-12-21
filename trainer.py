import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from utils import DiceLoss, region_loss
from utils import test_single_volume, calculate_clinical_metrics


def trainer_acdc(args, model, snapshot_path, optimizer, scheduler):
    from datasets.dataset_acdc import BaseDataSets, RandomGenerator
    
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epochs = args.max_epochs

    db_train = BaseDataSets(base_dir=args.root_path, list_dir=args.list_dir, split="train", 
                            transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]))
    db_test = BaseDataSets(base_dir=args.root_path, list_dir=args.list_dir, split="test")
    
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    best_performance = 0.0
    l1, l2, l3 = 0.45, 0.45, 0.1

    for epoch_num in range(max_epochs):
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            outputs = model(volume_batch)
    
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_region = region_loss(outputs, label_batch, num_classes=num_classes)

            loss = l1 * loss_ce + l2 * loss_dice + l3 * loss_region
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_num += 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            
            if iter_num % 50 == 0:
                logging.info('epoch %d iteration %d : loss : %f, ce: %f, dice: %f, region: %f' % 
                             (epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_region.item()))

        scheduler.step()
        
        if (epoch_num + 1) % 20 == 0:
            model.eval()
            metric_list = 0.0
            
            # 建立嵌套字典，用于按实体(Entity)存储所有样本的指标
            # 对应 ACDC 的 1:RV, 2:MYO, 3:LV
            entity_stats = {
                e: {'iou': [], 'precision': [], 'recall': []} 
                for e in range(1, num_classes)
            }
            
            for i_batch, sampled_batch in enumerate(testloader):
                image, label = sampled_batch["image"], sampled_batch["label"]
                
                # 1. 基础分割指标计算 (Dice, HD95)
                metric_i = test_single_volume(image, label, model, classes=num_classes, 
                                              patch_size=[args.img_size, args.img_size],
                                              z_spacing=1)
                metric_list += np.array(metric_i)

                clinical_output = calculate_clinical_metrics(
                    model, image, label, 
                    target_entities=[1, 2, 3], 
                    img_size=args.img_size,
                    eval_types=['iou', 'precision', 'recall'],
                    device='cuda'
                )
                
                for e in range(1, num_classes):
                    e_key = f'entity_{e}'
                    if e_key in clinical_output:
                        res = clinical_output[e_key]
                        entity_stats[e]['iou'].append(res['iou'])
                        entity_stats[e]['precision'].append(res['precision'])
                        entity_stats[e]['recall'].append(res['recall'])
            
            metric_list = metric_list / len(db_test)
            performance = np.mean(metric_list, axis=0)[0]
            
            logging.info('Epoch %d Detailed Validation Report:' % epoch_num)
            all_iou, all_prec, all_rec = [], [], []
            
            for e in range(1, num_classes):
                e_iou = np.mean(entity_stats[e]['iou'])
                e_prec = np.mean(entity_stats[e]['precision'])
                e_rec = np.mean(entity_stats[e]['recall'])
                
                all_iou.append(e_iou)
                all_prec.append(e_prec)
                all_rec.append(e_rec)
                
                logging.info('>> Entity %d | IoU: %.4f | Prec: %.4f | Rec: %.4f' % 
                             (e, e_iou, e_prec, e_rec))
            
            logging.info('>> SUMMARY | Mean Dice: %.4f | mIoU: %.4f | mPrec: %.4f | mRec: %.4f' % 
                         (performance, np.mean(all_iou), np.mean(all_prec), np.mean(all_rec)))

            if performance > best_performance:
                best_performance = performance
                torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))
                logging.info('New best model saved at epoch %d' % (epoch_num))

            model.train()

    writer.close()
    return "Training Finished!"


def trainer_mscmrseg(args, model, snapshot_path, optimizer, scheduler):
    
    from datasets.dataset_MSCMRSeg import MSCMRDataSets, RandomGenerator

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    num_classes = args.num_classes
    batch_size = args.batch_size
    max_epochs = args.max_epochs

    db_train = MSCMRDataSets(base_dir=args.root_path, split="train", 
                             transform=transforms.Compose([RandomGenerator([args.img_size, args.img_size])]))
    db_test = MSCMRDataSets(base_dir=args.root_path, split="test") 
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, 
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    best_performance = 0.0
    
    l1, l2, l3 = 0.45, 0.45, 0.1

    for epoch_num in range(max_epochs):
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            outputs = model(volume_batch)
            
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss_region = region_loss(outputs, label_batch, num_classes=num_classes)
            
            loss = l1 * loss_ce + l2 * loss_dice + l3 * loss_region
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_num += 1
            writer.add_scalar('info/total_loss', loss, iter_num)
            
            if iter_num % 50 == 0:
                logging.info('epoch %d iteration %d : loss : %f, ce: %f, dice: %f, region: %f' % 
                             (epoch_num, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), loss_region.item()))

        scheduler.step()

        if (epoch_num + 1) % 20 == 0:
            model.eval()
            metric_list = 0.0
            entity_stats = {
                e: {'iou': [], 'precision': [], 'recall': []} 
                for e in range(1, num_classes)
            }
            
            for i_batch, sampled_batch in enumerate(testloader):
                image, label = sampled_batch["image"], sampled_batch["label"]
                
                metric_i = test_single_volume(image, label, model, classes=num_classes, 
                                              patch_size=[args.img_size, args.img_size],
                                              z_spacing=1)
                metric_list += np.array(metric_i)

                clinical_output = calculate_clinical_metrics(
                    model, image, label, 
                    target_entities=[1, 2, 3], 
                    img_size=args.img_size,
                    eval_types=['iou', 'precision', 'recall'],
                    device='cuda'
                )
                
                for e in range(1, num_classes):
                    e_key = f'entity_{e}'
                    if e_key in clinical_output:
                        res = clinical_output[e_key]
                        entity_stats[e]['iou'].append(res['iou'])
                        entity_stats[e]['precision'].append(res['precision'])
                        entity_stats[e]['recall'].append(res['recall'])
            
            metric_list = metric_list / len(db_test)
            performance = np.mean(metric_list, axis=0)[0]
            
            logging.info('Epoch %d Detailed Validation Report (MSCMRSeg):' % epoch_num)
            all_iou, all_prec, all_rec = [], [], []
            
            for e in range(1, num_classes):
                e_iou = np.mean(entity_stats[e]['iou'])
                e_prec = np.mean(entity_stats[e]['precision'])
                e_rec = np.mean(entity_stats[e]['recall'])
                
                all_iou.append(e_iou)
                all_prec.append(e_prec)
                all_rec.append(e_rec)
                
                logging.info('>> Entity %d | IoU: %.4f | Prec: %.4f | Rec: %.4f' % 
                             (e, e_iou, e_prec, e_rec))
            
            logging.info('>> SUMMARY | Mean Dice: %.4f | mIoU: %.4f | mPrec: %.4f | mRec: %.4f' % 
                         (performance, np.mean(all_iou), np.mean(all_prec), np.mean(all_rec)))

            if performance > best_performance:
                best_performance = performance
                torch.save(model.state_dict(), os.path.join(snapshot_path, 'best_model.pth'))
                logging.info('New best model saved at epoch %d with dice: %f' % (epoch_num, performance))

            model.train()

    writer.close()
    return "MSCMRSeg Training Finished!"