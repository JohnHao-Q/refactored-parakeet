from .rgb_extact import Hourglass
from .depth_extact import Light_depth
from .ECA_model import eca_layer

import torch
import torch.nn.functional as F
from torch import nn

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

def Deconv(in_planes, out_planes):
    return nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1 , dilation=1, bias=True), nn.BatchNorm2d(in_planes), nn.ReLU(), nn.Conv2d(in_planes, in_planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True), nn.ConvTranspose2d(in_planes, out_planes,kernel_size=3,stride=2,padding=1,output_padding=1))


class New_net(nn.Module):


    def __init__(self, max_depth):
        super(New_net, self).__init__()
        self.MAX_DEPTH = max_depth
        self.rgb_extact = Hourglass(channel=3)
        self.depth_extact = Light_depth(channel=1)
        self.eca = eca_layer(channel=256, k_size=3)
        
        self.joint_conv1 = nn.Sequential(ConvBN(256, 128, 3, 1, 1, 1),nn.ReLU(), nn.Conv2d(128,128,kernel_size=1,padding=0,stride=1,bias=True))#128/8        
        self.deconv1 = Deconv(128, 64)#上采样64/4
        self.joint_conv2 = nn.Sequential(ConvBN(128,128,3,1,1,1),nn.ReLU(),nn.Conv2d(128,64,kernel_size=3,padding=1,stride=1,bias=True))
        
        self.deconv2 = Deconv(64, 32)
        self.joint_conv3 = nn.Sequential(ConvBN(64,64,3,1,1,1),nn.ReLU(),nn.Conv2d(64,32,kernel_size=3,padding=1,stride=1,bias=True))
        
        self.deconv3 = Deconv(32, 16)
        self.joint_conv4 = nn.Sequential(ConvBN(32,32,3,1,1,1),nn.ReLU(),nn.Conv2d(32,16,kernel_size=3,padding=1,stride=1,bias=True))
        
        self.mapping = nn.Sequential(ConvBN(16, 1, 3, 1, 1, 1),nn.ReLU(), nn.Conv2d(1,1,kernel_size=1,padding=0,stride=1,bias=True))
        
        self.final_conv = nn.Sequential(ConvBN(225,225,3,1,1,1),nn.ReLU(),nn.Conv2d(225,1,kernel_size=1,padding=0,stride= 1,bias=True))
        self.final_sigmoid = nn.Sigmoid()
        
        
                
    def forward(self, x_in):
        rgb_in = x_in[:,1:,:,:]
        d_in = torch.unsqueeze(x_in[:,0,:,:] / self.MAX_DEPTH, 1)
        
        map_128, map_64, map_32, map_16, map_1 = self.rgb_extact(rgb_in)
        d_128 = self.depth_extact(d_in)
        fuse_map = self.eca(torch.cat((map_128, d_128), dim=1))#256/8
        
        joint_feature1 = self.joint_conv1(fuse_map)#128/8
        deconv_1 = self.deconv1(joint_feature1)#64/4
        joint_feature2 = self.joint_conv2(torch.cat((map_64, deconv_1),1))#64/4
        deconv_2 = self.deconv2(joint_feature2)#32/2
        joint_feature3 = self.joint_conv3(torch.cat((map_32, deconv_2),1))#32/2
        deconv_3 = self.deconv3(joint_feature3)#16/1
        joint_feature4 = self.joint_conv4(torch.cat((map_16, deconv_3),1))#16/1
        
        depth_1 = self.mapping(joint_feature4)
        
        depth_8 = F.interpolate(input=joint_feature1,size =(depth_1.size()[2],depth_1.size()[3]),scale_factor=None,mode='nearest')
        depth_4 = F.interpolate(input=joint_feature2,size =(depth_1.size()[2],depth_1.size()[3]),scale_factor=None,mode='nearest')
        depth_2 = F.interpolate(input=joint_feature3,size =(depth_1.size()[2],depth_1.size()[3]),scale_factor=None,mode='nearest')
        
        decon_stack = torch.cat((depth_8,depth_4,depth_2,depth_1),dim=1)
        final_conv = self.final_conv(decon_stack)
        final_conv = self.final_sigmoid(final_conv)
        
        return final_conv* self.MAX_DEPTH, map_1* self.MAX_DEPTH
        
        
