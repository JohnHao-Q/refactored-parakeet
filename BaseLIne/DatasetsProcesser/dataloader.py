"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import numpy as np
from torch.utils.data import Dataset, DataLoader##本程序中的Dataset_loader为python对象，DataLoader为torch自带的迭代器
from torchvision import transforms, utils
import os
import torch
from PIL import Image
import random
import torchvision.transforms.functional as F
from Utils.utils import downsample


def get_loader(args, dataset):
    """
    Define the different dataloaders for training and validation
    """
    crop_size = (args.crop_h, args.crop_w)
    perform_transformation = not args.no_aug#图像不翻转时

    train_dataset = Dataset_loader(
            args.data_path, dataset.train_paths, args.input_type, resize=None,
            rotate=args.rotate, crop=crop_size, flip=args.flip, rescale=args.rescale,
            max_depth=args.max_depth, sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=perform_transformation, num_samples=args.num_samples)
    val_dataset = Dataset_loader(
            args.data_path, dataset.val_paths, args.input_type, resize=None,
            rotate=args.rotate, crop=crop_size, flip=args.flip, rescale=args.rescale,
            max_depth=args.max_depth, sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=False, num_samples=args.num_samples)
    val_select_dataset = Dataset_loader(
            args.data_path, dataset.selected_paths, args.input_type,
            resize=None, rotate=args.rotate, crop=crop_size,
            flip=args.flip, rescale=args.rescale, max_depth=args.max_depth,
            sparse_val=args.sparse_val, normal=args.normal, 
            disp=args.use_disp, train=False, num_samples=args.num_samples)

    train_sampler = None
    val_sampler = None
    if args.subset is not None:#子集操作
        random.seed(1)
        train_idx = [i for i in random.sample(range(len(train_dataset)-1), args.subset)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        #random.seed(1)
        #val_idx = [i for i in random.sample(range(len(val_dataset)-1), round(args.subset*0.5))]
        #val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=train_sampler is None, num_workers=args.nworkers,
        pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=int(args.val_batch_size),  sampler=val_sampler,
        shuffle=val_sampler is None, num_workers=args.nworkers_val,
        pin_memory=True, drop_last=True)
    val_selection_loader = DataLoader(
        val_select_dataset, batch_size=int(args.val_batch_size), shuffle=False,
        num_workers=args.nworkers_val, pin_memory=True, drop_last=True)
    return train_loader, val_loader, val_selection_loader


class Dataset_loader(Dataset):
    """Dataset with labeled lanes"""

    def __init__(self, data_path, dataset_type, input_type, resize,
                 rotate, crop, flip, rescale, max_depth, sparse_val=0.0, 
                 normal=False, disp=False, train=False, num_samples=None):

        # Constants
        self.use_rgb = input_type == 'rgb'##是否使用RGB
        self.datapath = data_path##数据路径
        self.dataset_type = dataset_type#数据集类型的具体路径标识
        self.train = train#是否翻转训练集图片
        self.resize = resize#调整大小
        self.flip = flip#翻转
        self.crop = crop#裁减大小
        self.rotate = rotate#旋转
        self.rescale = rescale#缩放
        self.max_depth = max_depth#最大深度
        self.sparse_val = sparse_val#填充值

        # Transformations
        self.totensor = transforms.ToTensor()#转为tensor
        self.center_crop = transforms.CenterCrop(size=crop)#中心剪裁

        # Names
        self.img_name = 'img'
        self.lidar_name = 'lidar_in' 
        self.gt_name = 'gt'
        self.ip_name = 'ip_gt' 

        # Define random sampler
        self.num_samples = num_samples


    def __len__(self):
        """
        Conventional len method
        """
        return len(self.dataset_type['lidar_in'])#深度数据多少即为多少长度


    def depth_read(self, img, sparse_val):
    # loads depth map D from png file and returns it as a numpy array, for details see readme.txt
    	depth_png = np.array(img, dtype=int)

    	depth_png = np.expand_dims(depth_png, axis=2)#numpy类型为H，W，C故在第二维度拓展方便后面和RGB拼接
    # make sure we have a proper 16bit depth map here.. not 8bit!
    	
    	assert(np.max(depth_png) > 255)
    	depth = depth_png.astype(np.float) / 6553.5
    	depth[depth_png == 0] = sparse_val
    	
    	return depth
    
    def define_transforms(self, input, gt, img=None, ip_gt=None):
        # Define random variabels#此处为数据增强操作函数，翻转裁减输入数据
        hflip_input = np.random.uniform(0.0, 1.0) > 0.5 and self.flip == 'hflip'##0-1随机取值，在要求翻转时50%hflip_input翻转，

        if self.train:#训练集翻转
            i, j, h, w = transforms.RandomCrop.get_params(input, output_size=self.crop)#随机裁减的中心位置i，j,长宽h，w
            input = F.crop(input, i, j, h, w)
            gt = F.crop(gt, i, j, h, w)
            if hflip_input:
                input, gt = F.hflip(input), F.hflip(gt)#水平翻转

            if self.use_rgb:
                img = F.crop(img, i, j, h, w)
                if hflip_input:
                    img = F.hflip(img)
                    
            if len(self.dataset_type) == 4 and False:
                ip_gt = F.crop(ip_gt, i, j, h, w)
                if hflip_input:
                    ip_gt = F.hflip(ip_gt) 
                       
            input, gt = self.depth_read(input, self.sparse_val), self.depth_read(gt, self.sparse_val)#读取深度操作
            
        else:
            input, gt = self.center_crop(input), self.center_crop(gt)
            if self.use_rgb:
                img = self.center_crop(img)
                
            if len(self.dataset_type) == 4 and False:
                ip_gt = self.center_crop(ip_gt) 
                
            input, gt = self.depth_read(input, self.sparse_val), self.depth_read(gt, self.sparse_val)#读取深度操作
            

        return input, gt, img

    def __getitem__(self, idx):
        """
        Args: idx (int): Index of images to make batch
        Returns (tuple): Sample of velodyne data and ground truth.
        """
        sparse_depth_name = os.path.join(self.dataset_type[self.lidar_name][idx])
        gt_name = os.path.join(self.dataset_type[self.gt_name][idx])     
        with open(sparse_depth_name, 'rb') as f:
            sparse_depth = Image.open(f)
            w, h = sparse_depth.size
            sparse_depth = F.crop(sparse_depth, h-self.crop[0], 0, self.crop[0], w)
        with open(gt_name, 'rb') as f:
            gt = Image.open(f)
            gt = F.crop(gt, h-self.crop[0], 0, self.crop[0], w)
        
        img = None
        if self.use_rgb:
            img_name = self.dataset_type[self.img_name][idx]
            with open(img_name, 'rb') as f:
                img = (Image.open(f).convert('RGB'))#从四通道转为RGB
            img = F.crop(img, h-self.crop[0], 0, self.crop[0], w)
        
        ip_gt = None     
        if len(self.dataset_type) == 4 and False:
            ip_name =os.path.join(self.dataset_type[self.ip_name][idx])
            with open(ip_name, 'rb') as f:
                ip_gt = Image.open(f)
                ip_gt = F.crop(ip_gt, h-self.crop[0], 0, self.crop[0], w)
                
        sparse_depth_np, gt_np, img_pil = self.define_transforms(sparse_depth, gt, img, ip_gt)#取出增强后的数据
        sparse_depth_np = downsample(sparse_depth_np, 4000)
        input, gt = self.totensor(sparse_depth_np).float(), self.totensor(gt_np).float()#将数据转为tensor,numpy通道维度位置与tensor不同

        if self.use_rgb:
            img_tensor = self.totensor(img_pil).float()
            input = torch.cat((input, img_tensor), dim=0)#拼接Depth和Rgb
        return input, gt#返回最后的数据集tensor

