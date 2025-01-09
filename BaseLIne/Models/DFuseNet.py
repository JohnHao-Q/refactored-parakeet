# ******************************************************************************
#                               DFuseNet
#
#    Shreyas S. Shivakumar, Ty Nguyen, Steven W.Chen and Camillo J. Taylor
#
#                               ( 2018 )
#
# 	This code has been written Shreyas S. Shivakumar and Ty Nguyen
#
#       University of Pennsylvania | {sshreyas,tynguyen}@seas.upenn.edu
#
# ******************************************************************************
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import collections
import math

import pdb
import time

def ConvBN(in_planes, out_planes, kernel_size, stride, pad, dilation):
    """
    Perform 2D Convolution with Batch Normalization
    """
    return nn.Sequential(nn.Conv2d(in_planes,
                                   out_planes,
                                   kernel_size = kernel_size,
                                   stride = stride,
                                   padding = dilation if dilation > 1 else pad,
                                   dilation = dilation,
                                   bias=True),
                         nn.BatchNorm2d(out_planes))

class BasicResBlock(nn.Module):
    """
    Basic Convolution block with Residual
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicResBlock, self).__init__()
        self.conv1 = ConvBN(inplanes, planes, 3, stride, pad, dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = ConvBN(planes, planes, 3, 1, pad, dilation)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out

class FeatureExtractRGB(nn.Module):
    """
    Feature extraction block for RGB branch
    """
    def __init__(self):
        super(FeatureExtractRGB, self).__init__()
        self.inplanes = 32

        self.fe_conv1 = ConvBN(3,32,3,2,1,1)
        self.fe_relu1 = nn.ReLU()
        self.fe_conv2 = ConvBN(32,32,3,1,1,1)
        self.fe_relu2 = nn.ReLU()

        self.fe_conv3_4   = self._make_layer(BasicResBlock,32,3,1,1,1)
        self.fe_conv5_8   = self._make_layer(BasicResBlock,64,16,2,1,1)
        self.fe_conv9_10  = self._make_layer(BasicResBlock,128,3,1,1,1)
        self.fe_conv11_12 = self._make_layer(BasicResBlock,128,3,1,1,2)

        self.level64_pool = nn.AvgPool2d((64, 64), stride=(64,64))
        self.level64_conv = ConvBN(128, 32, 1, 1, 0, 1)
        self.level64_relu = nn.ReLU()

        self.level32_pool = nn.AvgPool2d((32, 32), stride=(32,32))
        self.level32_conv = ConvBN(128, 32, 1, 1, 0, 1)
        self.level32_relu = nn.ReLU()

        self.level16_pool = nn.AvgPool2d((16, 16), stride=(16,16))
        self.level16_conv = ConvBN(128, 32, 1, 1, 0, 1)
        self.level16_relu = nn.ReLU()

        self.level8_pool = nn.AvgPool2d((8, 8), stride=(8,8))
        self.level8_conv = ConvBN(128, 32, 1, 1, 0, 1)
        self.level8_relu = nn.ReLU()

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)
        layers = []
        layers.append(block(self.inplanes,
                            planes,
                            stride,
                            downsample,
                            pad,
                            dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        op_conv1 = self.fe_conv1(x)
        op_relu1 = self.fe_relu1(op_conv1)
        op_conv2 = self.fe_conv2(op_relu1)
        op_relu2 = self.fe_relu2(op_conv2)

        op_conv3_4   = self.fe_conv3_4(op_relu2)
        op_conv5_8   = self.fe_conv5_8(op_conv3_4)
        op_conv9_10  = self.fe_conv9_10(op_conv5_8)
        op_conv11_12 = self.fe_conv11_12(op_conv9_10)

        interp_size = (op_conv11_12.size()[2], op_conv11_12.size()[3])

        op_l64_pool     = self.level64_pool(op_conv11_12)
        op_l64_conv     = self.level64_conv(op_l64_pool)
        op_l64          = self.level64_relu(op_l64_conv)
        op_l64_upsample = F.interpolate(input = op_l64,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='bilinear',
                                      align_corners=True)

        op_l32_pool     = self.level32_pool(op_conv11_12)
        op_l32_conv     = self.level32_conv(op_l64_pool)
        op_l32          = self.level32_relu(op_l64_conv)
        op_l32_upsample = F.interpolate(input = op_l32,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='bilinear',
                                      align_corners=True)

        op_l16_pool     = self.level16_pool(op_conv11_12)
        op_l16_conv     = self.level16_conv(op_l16_pool)
        op_l16          = self.level16_relu(op_l16_conv)
        op_l16_upsample = F.interpolate(input = op_l16,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='bilinear',
                                      align_corners=True)

        op_l8_pool      = self.level8_pool(op_conv11_12)
        op_l8_conv      = self.level8_conv(op_l8_pool)
        op_l8           = self.level8_relu(op_l8_conv)
        op_l8_upsample = F.interpolate(input = op_l8,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='bilinear',
                                      align_corners=True)

        return op_conv5_8, op_conv11_12, op_l8_upsample, op_l16_upsample, op_l32_upsample, op_l64_upsample


class FeatureExtractDepth(nn.Module):
    """
    Feature extraction block for Depth branch
    """
    def __init__(self):
        super(FeatureExtractDepth, self).__init__()
        self.inplanes = 32

        self.conv_block1 = nn.Sequential(ConvBN(1, 16, 11, 1, 5, 1),
                                         nn.ReLU())
        self.conv_block2 = nn.Sequential(ConvBN(16, 32, 7, 2, 3, 1),
                                         nn.ReLU())
        self.conv_block3 = nn.Sequential(ConvBN(32, 64, 5, 2, 2, 1),
                                         nn.ReLU())

        self.level64_pool = nn.MaxPool2d((64, 64), stride=(64,64))
        self.level64_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level64_relu = nn.ReLU()

        self.level32_pool = nn.MaxPool2d((32, 32), stride=(32,32))
        self.level32_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level32_relu = nn.ReLU()

        self.level16_pool = nn.MaxPool2d((16, 16), stride=(16,16))
        self.level16_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level16_relu = nn.ReLU()

        self.level8_pool = nn.MaxPool2d((8, 8), stride=(8,8))
        self.level8_conv = ConvBN(64, 32, 1, 1, 0, 1)
        self.level8_relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        m_in = (x > 0).detach().float()
        new_conv1 = self.conv_block1(x)
        new_conv2 = self.conv_block2(new_conv1)
        new_conv3 = self.conv_block3(new_conv2)
        interp_size = (new_conv3.size()[2], new_conv3.size()[3])
        op_maskconv = new_conv3

        op_l64_pool     = self.level64_pool(op_maskconv)
        op_l64_conv     = self.level64_conv(op_l64_pool)
        op_l64          = self.level64_relu(op_l64_conv)
        op_l64_upsample = F.interpolate(input = op_l64,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')

        op_l32_pool     = self.level32_pool(op_maskconv)
        op_l32_conv     = self.level32_conv(op_l64_pool)
        op_l32          = self.level32_relu(op_l64_conv)
        op_l32_upsample = F.interpolate(input = op_l32,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')

        op_l16_pool     = self.level16_pool(op_maskconv)
        op_l16_conv     = self.level16_conv(op_l16_pool)
        op_l16          = self.level16_relu(op_l16_conv)
        op_l16_upsample = F.interpolate(input = op_l16,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')

        op_l8_pool      = self.level8_pool(op_maskconv)
        op_l8_conv      = self.level8_conv(op_l8_pool)
        op_l8           = self.level8_relu(op_l8_conv)
        op_l8_upsample = F.interpolate(input = op_l8,
                                      size = interp_size,
                                      scale_factor=None,
                                      mode='nearest')
        return op_maskconv, op_l8_upsample, op_l16_upsample, op_l32_upsample, op_l64_upsample



class DFuseNet(nn.Module):
    def __init__(self, max_depth):
        super(DFuseNet, self).__init__()
        self.KITTI_MAX_DEPTH = max_depth
        self.FeatureExtractRGB = FeatureExtractRGB()
        self.FeatureExtractDepth = FeatureExtractDepth()

        # Joint RGB and Depth layers
        self.joint_conv1 = nn.Sequential(ConvBN(320, 192, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(192, 192, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True),
                                       nn.ReLU())

        self.joint_conv2 = nn.Sequential(ConvBN(192, 192, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(192, 192, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True),
                                       nn.ReLU())

        self.joint_conv3 = nn.Sequential(ConvBN(192, 128, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(128, 128, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))

        # Deconvolution / Reconstruction Layers
        self.deconv_l1 = nn.ConvTranspose2d(128,
                                            128,
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            dilation = 1)#*2
        self.deconv_l2 = nn.Sequential(ConvBN(128, 64, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(64, 64, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))
        self.deconv_l3 = nn.ConvTranspose2d(64,
                                            64,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 1,
                                            dilation = 1)#*1
        self.deconv_l4 = nn.Sequential(ConvBN(64, 32, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(32, 32, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))
        self.deconv_l5 = nn.ConvTranspose2d(32,
                                            32,
                                            kernel_size = 3,
                                            stride = 2,
                                            padding = 1,
                                            dilation = 1)#*2
        self.deconv_l6 = nn.Sequential(ConvBN(32, 16, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(16, 16, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))
        self.deconv_l7 = nn.ConvTranspose2d(16,
                                            16,
                                            kernel_size = 3,
                                            stride = 1,
                                            padding = 1,
                                            dilation = 1)#*1
        self.deconv_l8 = nn.Sequential(ConvBN(16, 1, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(1, 1, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))

        self.d_s1_depth = nn.Sequential(ConvBN(128, 1, 1, 1, 0, 1),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(1, 1, kernel_size = 1,
                                            padding=0, stride = 1,
                                                  bias = True))

        self.d_s2_depth = nn.Sequential(ConvBN(64, 1, 1, 1, 0, 1),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(1, 1, kernel_size = 1,
                                                  padding=0, stride = 1,
                                                  bias = True))


        self.d_s3_depth = nn.Sequential(ConvBN(32, 1, 1, 1, 0, 1),
                                        nn.LeakyReLU(0.2),
                                        nn.Conv2d(1, 1, kernel_size = 1,
                                                  padding=0, stride = 1,
                                                  bias = True))

        self.final_conv = nn.Sequential(ConvBN(113, 113, 3, 1, 1, 1),
                                       nn.ReLU(),
                                       nn.Conv2d(113, 1, kernel_size=1,
                                                 padding=0, stride = 1,
                                                 bias=True))
        self.final_sigmoid = nn.Sigmoid()

    def forward(self, x_in):

        rgb_in = x_in[:,1:,:,:]
        d_in = x_in[:,0,:,:] / self.KITTI_MAX_DEPTH
        i_raw, i_skip, i_b4, i_b3, i_b2, i_b1 = self.FeatureExtractRGB(rgb_in)

        d_skip, d_b4, d_b3, d_b2, d_b1 = self.FeatureExtractDepth(d_in)

        id_raw_cat = torch.cat((i_raw,
                                d_b4,
                                i_b4,
                                d_b3,
                                i_b3,
                                d_b2,
                                i_b2,
                                d_b1,
                                i_b1), 1)

        jf_s1 = self.joint_conv1(id_raw_cat)
        jf_s2 = self.joint_conv2(jf_s1)
        jf_s3 = self.joint_conv3(jf_s2)

        decon_1 = self.deconv_l1(jf_s3, output_size=(x_in.size()[2]//2,
                                                     x_in.size()[3]//2))
        decon_2 = self.deconv_l2(decon_1)
        decon_3 = self.deconv_l3(decon_2, output_size=(x_in.size()[2]//2,
                                                       x_in.size()[3]//2))
        decon_4 = self.deconv_l4(decon_3)
        decon_5 = self.deconv_l5(decon_4, output_size=(x_in.size()[2],
                                                       x_in.size()[3]))
        decon_6 = self.deconv_l6(decon_5)
        decon_7 = self.deconv_l7(decon_6, output_size=(x_in.size()[2],
                                                       x_in.size()[3]))
        decon_8 = self.deconv_l8(decon_7)

        decon_2_l1 = F.interpolate(input = decon_2,
                                  size = (decon_8.size()[2],
                                          decon_8.size()[3]),
                                  scale_factor=None,
                                  mode='nearest')
        decon_4_l1 = F.interpolate(input = decon_4,
                                  size = (decon_8.size()[2],
                                          decon_8.size()[3]),
                                  scale_factor=None)
        decon_6_l1 = F.interpolate(input = decon_6,
                                  size = (decon_8.size()[2],
                                          decon_8.size()[3]),
                                  scale_factor=None,
                                  mode='nearest')
        decon_8_l1 = F.interpolate(input = decon_8,
                                  size = (decon_8.size()[2],
                                          decon_8.size()[3]),
                                  scale_factor=None,
                                  mode='nearest')
        decon_stack = torch.cat((decon_2_l1,
                                 decon_4_l1,
                                 decon_6_l1,
                                 decon_8_l1), 1)
        final_conv = self.final_conv(decon_stack)
        final_conv = self.final_sigmoid(final_conv) * self.KITTI_MAX_DEPTH
        return final_conv
