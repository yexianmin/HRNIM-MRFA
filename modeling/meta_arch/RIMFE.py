import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        groups = math.gcd(in_channels, out_channels)
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

class HCMCFE(nn.Module):
    def __init__(self, distill = True):
        super(HCMCFE, self).__init__()
        self.down1_t = nn.Upsample(scale_factor=1/2, mode='nearest')
        self.down1_i = nn.Upsample(scale_factor=1/2, mode='bilinear')
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.distill = distill
        
        self.w1 = 16/21
        self.w2 = 4/21
        self.w3 = 1/21
        
        self.fea_ex0_1 = nn.Sequential(DoubleConv(4, 4),
                                       nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1, groups = 4))
        self.fea_ex1_1 = nn.Sequential(DoubleConv(8, 8),
                                       nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, groups = 4))
        self.fea_ex2_1 = nn.Sequential(DoubleConv(16, 16),
                                       nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, groups = 4))
        
        self.fea_ex0_2 = nn.Sequential(DoubleConv(4, 4),
                                       nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1, groups = 4))
        self.fea_ex1_2 = nn.Sequential(DoubleConv(8, 8),
                                       nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, groups = 4))
        self.fea_ex2_2 = nn.Sequential(DoubleConv(16, 16),
                                       nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, groups = 4))
        
        self.fea_ex0_4 = nn.Sequential(DoubleConv(4, 4),
                                       nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1, groups = 4))
        self.fea_ex1_4 = nn.Sequential(DoubleConv(8, 8),
                                       nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, groups = 4))
        self.fea_ex2_4 = nn.Sequential(DoubleConv(16, 16),
                                       nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, groups = 4))
        
        
        self.fea_fuison_layer1 = nn.Sequential(DoubleConv(8, 8),
                                               nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False, groups = 4),
                                               nn.BatchNorm2d(16),
                                               nn.LeakyReLU(0.2, inplace=True))
        self.fea_fuison_layer2 = nn.Sequential(DoubleConv(16, 16),
                                               nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False, groups = 4),
                                               nn.BatchNorm2d(32),
                                               nn.LeakyReLU(0.2, inplace=True))
        self.fea_fuison_layer3 = nn.Sequential(DoubleConv(32, 32),
                                               nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False, groups = 4),
                                               nn.BatchNorm2d(64),
                                               nn.LeakyReLU(0.2, inplace=True))
        
        self.fea_fuison_layer4 = nn.Sequential(DoubleConv(64, 32),
                                               DoubleConv(32, 16),
                                               DoubleConv(16, 8),
                                               DoubleConv(8, 4),
                                               DoubleConv(4, 1))

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beita = nn.Parameter(torch.tensor(0.1))
        self.gama = nn.Parameter(torch.tensor(0.5))


    def forward(self, image):
        # (64/85) (16/85) (4/85) (1/85)
        # (16/21) (4/21) (1/21)
        # (4/5) (1/5)
        # trimap = image[:, 3:4, :, :]
        # trimap = torch.abs(torch.round(trimap) - trimap) * 2
        # image = image[:, 0:3, :, :]
        
        image_2 = self.down1_t(image)
        # trimap_2 = self.down1_t(trimap)
        
        image_4 = self.down1_t(image_2)
        # trimap_4 = self.down1_t(trimap_2)
        
        # image, image_2, image_4 = image * trimap, image_2 * trimap_2, image_4 * trimap_4
        
        fea0, fea1, fea2 = self.fea_ex0_1(image), self.fea_ex0_2(image_2), self.fea_ex0_4(image_4)
        if self.training:
            res_loss0 = (F.mse_loss(fea0, self.up(fea1)) + F.mse_loss(fea0, self.up1(fea2)) + F.mse_loss(fea1, self.up(fea2))) / 3
        fea_fus0 = self.w1 * fea0 + self.w2 * self.up(fea1) + self.w3 * self.up1(fea2) # [B, 6, H/2, W/2]
        
        fea0, fea1, fea2 = self.fea_ex1_1(fea0), self.fea_ex1_2(fea1), self.fea_ex1_4(fea2)
        if self.training:
            res_loss1 = (F.mse_loss(fea0, self.up(fea1)) + F.mse_loss(fea0, self.up1(fea2)) + F.mse_loss(fea1, self.up(fea2))) / 3
        fea_fus1 = self.w1 * fea0 + self.w2 * self.up(fea1) + self.w3 * self.up1(fea2) #[B, 12, H/4, W/4]
        
        fea0, fea1, fea2 = self.fea_ex2_1(fea0), self.fea_ex2_2(fea1), self.fea_ex2_4(fea2)
        if self.training:
            res_loss2 = (F.mse_loss(fea0, self.up(fea1)) + F.mse_loss(fea0, self.up1(fea2)) + F.mse_loss(fea1, self.up(fea2))) / 3
        fea_fus2 = self.w1 * fea0 + self.w2 * self.up(fea1) + self.w3 * self.up1(fea2) # [B, 24, H/8, W/8]
        
        fea_fus = self.fea_fuison_layer1(fea_fus0) + fea_fus1
        fea_fus = self.fea_fuison_layer2(fea_fus) + fea_fus2
        fea_fus = self.fea_fuison_layer3(fea_fus)
        fea_fus = self.fea_fuison_layer4(fea_fus)
        fea_fus = self.gama * torch.tanh(self.alpha * fea_fus) + self.beita
                
        if self.training:
            res_loss = (res_loss0 + res_loss1 + res_loss2) / 3
            return fea_fus, res_loss
        else:
            return fea_fus

# class Matting_head(nn.Module):
#     def __init__(self):
#         super(Matting_head, self).__init__()
#         self.up = nn.Sequential(
#             nn.ConvTranspose2d(3, 6, kernel_size=3, stride=2, padding=1, output_padding=1, groups=3),
#             nn.Conv2d(6, 3, kernel_size=3, padding=1, bias=False, groups=3),
#             nn.BatchNorm2d(3),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=3))
        
#     def forward(self, outs):
#         outs = self.up(outs)
#         outs = (torch.tanh(outs) + 1.0) / 2.0
#         return outs

