import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

def region_loss(outputs, targets, num_classes=9):
    
    probs = F.softmax(outputs, dim=1)
    batch_size, channels, h, w = probs.size()
    loss_e = 0.0
    k1, k2 = 1.0, 0.01
    valid_entities = 0

    for b in range(batch_size):
        sample_probs = probs[b]    
        sample_target = targets[b]    

        for e in range(1, num_classes):
            mask = (sample_target == e)
            ne = torch.sum(mask)
            
            if ne > 0:
                ke = torch.sum(sample_probs * mask, dim=(1, 2)) / ne
                ke = ke.unsqueeze(0) # (1, C)
                
                target_oh = torch.zeros((1, num_classes), device=ke.device)
                target_oh[0, e] = 1.0
                
                l_ce = F.binary_cross_entropy(ke, target_oh) 
                l_mse = F.mse_loss(ke, target_oh)
                
                loss_e += (k1 * l_ce + k2 * l_mse)
                valid_entities += 1
                
    return loss_e / max(valid_entities, 1)

def calculate_clinical_metrics(model, image, label, target_entities=[1, 2, 3], 
                               img_size=224, eval_types=['iou', 'precision', 'recall'], device='cuda'):
    model.eval()
    with torch.no_grad():
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        outputs = model(image.to(device))
        preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0).cpu().numpy()
        gt = label.squeeze().cpu().numpy()
        
        results = {}
        
        for entity in target_entities:
            entity_results = {}
            pred_mask = (preds == entity)
            gt_mask = (gt == entity)
            iou,precision,recall = 0,0,0
            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()
            if 'iou' in eval_types:
                # entity_results['iou'] = (intersection + 1e-6) / (union + 1e-6)
                entity_results['iou'] = iou
            if 'precision' in eval_types:
                entity_results['precision'] = precision
            if 'recall' in eval_types:
                entity_results['recall'] = recall
                
            results[f'entity_{entity}'] = entity_results

    return results

def get_clinical_metrics_summary(clinical_records):
    if not clinical_records or len(clinical_records['iou']) == 0:
        return "No clinical metrics recorded."
    
    mean_iou = np.mean(clinical_records['iou'])
    mean_prec = np.mean(clinical_records['precision'])
    mean_rec = np.mean(clinical_records['recall'])
    
    return f"mIoU: {mean_iou:.4f}, Precision: {mean_prec:.4f}, Recall: {mean_rec:.4f}"

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list