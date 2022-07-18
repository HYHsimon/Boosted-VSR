import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from networks.correlation import correlation  # the custom cost volume layer
except:
    sys.path.insert(0, './correlation');
    import correlation  # you should consider upgrading python

from networks.base_networks import DCNv2Pack_test

class PCD_Corr_Align(nn.Module):
    ''' Alignment module using Pyramid, Cascading and Deformable convolution
    with 3 pyramid levels.
    '''

    def __init__(self, nf=64, groups=8):
        super(PCD_Corr_Align, self).__init__()
        print('Now Initializing PCD_Corr_Align_v2_test...')
        # L3: level 3, 1/4 spatial size
        self.L3_offset_conv1 = nn.Conv2d(nf + 81, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L3_dcnpack = DCNv2Pack_test(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        # L2: level 2, 1/2 spatial size
        self.L2_offset_conv1 = nn.Conv2d(nf + 81, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L2_dcnpack = DCNv2Pack_test(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.L1_offset_conv1 = nn.Conv2d(nf + 81, nf, 3, 1, 1, bias=True)  # concat for diff
        self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.L1_dcnpack = DCNv2Pack_test(nf, nf, 3, stride=1, padding=1, dilation=1,
                                  deformable_groups=groups)
        self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf + 81, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_dcnpack = DCNv2Pack_test(nf, nf, 3, stride=1, padding=1, dilation=1,
                                   deformable_groups=groups)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l, last_fea_l):
        '''align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        '''
        # L3
        tensorVolume = self.lrelu(
            input=correlation.FunctionCorrelation(
                tensorFirst=F.interpolate(ref_fea_l[2], scale_factor=0.25, mode='bilinear'),
                tensorSecond=F.interpolate(nbr_fea_l[2], scale_factor=0.25, mode='bilinear'),
            ))
        L3_offset = torch.cat([F.interpolate(tensorVolume, scale_factor=4, mode='nearest'),
                               ref_fea_l[2]], dim=1)
        # L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
        L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
        L3_fea, L3_offset_absmean = self.L3_dcnpack(last_fea_l[2], L3_offset)
        L3_fea = self.lrelu(L3_fea)
        # L2
        tensorVolume = self.lrelu(
            input=correlation.FunctionCorrelation(
                tensorFirst=F.interpolate(ref_fea_l[1], scale_factor=0.125, mode='bilinear'),
                tensorSecond=F.interpolate(nbr_fea_l[1], scale_factor=0.125, mode='bilinear'),
            ))
        L2_offset = torch.cat([F.interpolate(tensorVolume, scale_factor=8, mode='nearest'),
                               ref_fea_l[1]], dim=1)
        # L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
        L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear')
        L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
        L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
        L2_fea, L2_offset_absmean = self.L2_dcnpack(last_fea_l[1], L2_offset)
        L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear')
        L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
        # L1
        tensorVolume = self.lrelu(
            input=correlation.FunctionCorrelation(
                tensorFirst=F.interpolate(ref_fea_l[0], scale_factor=0.0625, mode='bilinear'),
                tensorSecond=F.interpolate(nbr_fea_l[0], scale_factor=0.0625, mode='bilinear'),
            ))
        L1_offset = torch.cat([F.interpolate(tensorVolume, scale_factor=16, mode='nearest'),
                               ref_fea_l[0]], dim=1)
        # L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
        L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear')
        L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
        L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
        L1_fea, L1_offset_absmean = self.L1_dcnpack(last_fea_l[0], L1_offset)
        L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear')
        L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
        # Cascading
        tensorVolume = self.lrelu(
            input=correlation.FunctionCorrelation(
                tensorFirst=F.interpolate(ref_fea_l[0], scale_factor=0.0625, mode='bilinear'),
                tensorSecond=F.interpolate(L1_fea, scale_factor=0.0625, mode='bilinear'),
            ))
        offset = torch.cat([F.interpolate(tensorVolume, scale_factor=16, mode='nearest'),
                               ref_fea_l[0]], dim=1)
        # offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
        offset = self.lrelu(self.cas_offset_conv1(offset))
        offset = self.lrelu(self.cas_offset_conv2(offset))
        L1_fea, cas_offset_absmean = self.cas_dcnpack(L1_fea, offset)
        L1_fea   = self.lrelu(L1_fea)

        return L1_fea, L3_offset_absmean, L2_offset_absmean, L1_offset_absmean, cas_offset_absmean