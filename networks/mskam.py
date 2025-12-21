import torch
import torch.nn as nn
from functools import partial
import math
from timm.layers import trunc_normal_tf_
from timm.models import named_apply
from .kantransformer import KAN as DGRKAN

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        return nn.ReLU(inplace)
    elif act == 'relu6':
        return nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        return nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'hswish':
        return nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x

class CAM(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio: 
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.m1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.m2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        self.gl = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.m2(self.gl(self.m1(self.avg_pool(x))))
        max_out = self.m2(self.gl(self.m1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out) * x

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        res = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(res)) * x

class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[1, 3, 5], activation='gelu'):
        super(MSDC, self).__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, k, padding=k // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                act_layer(activation)
            ) for k in kernel_sizes
        ])

    def forward(self, x):
        outputs = []
        for dw in self.dwconvs:
            outputs.append(dw(x))
        out = sum(outputs)
        return channel_shuffle(out, gcd(self.in_channels, self.in_channels))

class MKAP(nn.Module):
    def __init__(self, in_channels):
        super(MKAP, self).__init__()
        self.in_channels = in_channels
        self.dgrkan1 = DGRKAN(in_channels, use_conv=True, act_init='gelu')
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.msdc = MSDC(in_channels)
        self.dgrkan2 = DGRKAN(in_channels, use_conv=True, act_init='gelu')
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.gl = nn.GELU()

    def forward(self, x):
        x = self.bn1(self.gl(self.dgrkan1(x)))
        x = self.dgrkan2(self.msdc(x))
        return self.bn2(x)

class MSKAM(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=16):
        super(MSKAM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.pae_cam = CAM(in_channels, ratio=ratio)
        self.pae_sam = SAM(kernel_size=7)
        
        self.dafm_conv = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)
        self.dafm_bn = nn.BatchNorm2d(in_channels)
        self.dafm_gl = nn.GELU()
        
        self.mkap = MKAP(in_channels)
        
        self.project = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x_cam = self.pae_cam(x)
        x_sam = self.pae_sam(x)
        
        x_cat = torch.cat([x_cam, x_sam], dim=1)
        x_dafm = self.dafm_gl(self.dafm_bn(self.dafm_conv(x_cat)))
        
        x_mkap = self.mkap(x_dafm)
        
        out = self.project(x_mkap)
        return out

if __name__ == '__main__':
    # (B,C,H,W)
    x = torch.randn(1, 64, 224, 224)

    
    Spatial_attention = SAM() # (B, C, H, W) -> (B, C, H, W)
    Channel_attention = CAM(in_channels=64) #  (B, C, H, W) -> (B, C, H, W)
    Model = MSKAM(in_channels=64, out_channels=64) # (B, C, H, W) -> (B, C, H, W)

    # Multi-scale convolutional attention module (MSKAM)
    x = Channel_attention(x) * x  
    x = Spatial_attention(x) * x  
    out = Model(x) 
    print(out.shape)