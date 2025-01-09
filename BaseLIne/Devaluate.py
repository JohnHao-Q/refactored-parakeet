"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import numpy as np
import os
import sys
import time
import shutil
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim
import Models
import Datasets
import warnings
import random
from Utils.utils import str2bool, AverageMeter, depth_read 
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime
from Loss.loss import define_loss, allowed_losses, MSE_loss
from Loss.benchmark_metrics import Metrics, allowed_metrics
from Datasets.dataloader import get_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Utils.utils import str2bool, define_optim, define_scheduler, \
                        Logger, AverageMeter, first_run, mkdir_if_missing, \
                        define_init_weights, init_distributed_mode

# Training setttings
parser = argparse.ArgumentParser(description='KITTI Depth Completion Task')
parser.add_argument('--dataset', type=str, default='kitti', choices=Datasets.allowed_datasets(), help='dataset to work with')#数据集位置
parser.add_argument('--nepochs', type=int, default=10, help='Number of epochs for training')#训练次数
parser.add_argument('--thres', type=int, default=0, help='epoch for pretraining')#预训练次数
parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch number for training')#起始训练次数
parser.add_argument('--mod', type=str, default='New_net', choices=Models.allowed_models(), help='Model for use')#训练模型
parser.add_argument('--batch_size', type=int, default=2, help='batch size')#批量大小
parser.add_argument('--val_batch_size', default=None, help='batch size selection validation set')#验证批量大小
parser.add_argument('--learning_rate', metavar='lr', type=float, default=1e-3, help='learning rate')#学习率
parser.add_argument('--no_cuda', action='store_true', help='no gpu usage')#GPU使用情况

parser.add_argument('--evaluate', action='store_true', help='only evaluate')#仅仅评估用
parser.add_argument('--resume', type=str, default='', help='resume latest saved run')#恢复最新的运行
parser.add_argument('--nworkers', type=int, default=8, help='num of threads')#线程
parser.add_argument('--nworkers_val', type=int, default=0, help='num of threads')#验证线程
parser.add_argument('--no_dropout', action='store_true', help='no dropout in network')#无权重衰减
parser.add_argument('--subset', type=int, default=None, help='Take subset of train set')#训练集子集
parser.add_argument('--input_type', type=str, default='rgb', choices=['depth','rgb'], help='use rgb for rgbdepth')#输入类型
parser.add_argument('--side_selection', type=str, default='image_03', help='train on one specific stereo camera')#相机特定选择
parser.add_argument('--no_tb', type=str2bool, nargs='?', const=True,
                    default=True, help="use mask_gt - mask_input as final mask for loss calculation")#结果选择误差输出
parser.add_argument('--test_mode', action='store_true', help='Do not use resume')#测试模型
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=True, help='use pretrained model')#使用预训练模型
parser.add_argument('--load_external_mod', type=str2bool, nargs='?', const=True, default=False, help='path to external mod')#其他模型路径

# Data augmentation settings
parser.add_argument('--crop_w', type=int, default=1216, help='width of image after cropping')#图像宽
parser.add_argument('--crop_h', type=int, default=256, help='height of image after cropping')#图像高
parser.add_argument('--max_depth', type=float, default=85.0, help='maximum depth of LIDAR input')#最大深度
parser.add_argument('--sparse_val', type=float, default=0.0, help='value to endode sparsity with')#填充值
parser.add_argument("--rotate", type=str2bool, nargs='?', const=True, default=False, help="rotate image")#图像旋转？
parser.add_argument("--flip", type=str, default='hflip', help="flip image: vertical|horizontal")#水平或垂直
parser.add_argument("--rescale", type=str2bool, nargs='?', const=True,
                    default=False, help="Rescale values of sparse depth input randomly")#深度输入随机缩放
parser.add_argument("--normal", type=str2bool, nargs='?', const=True, default=True, help="normalize depth/rgb input")#归一化输入
parser.add_argument("--no_aug", type=str2bool, nargs='?', const=True, default=False, help="rotate image")#图像旋转？

# Paths settings
parser.add_argument('--save_path', default='./', help='save path')#保存路径
parser.add_argument('--data_path', required=True, help='path to desired dataset')#读取路径

# Optimizer settings
parser.add_argument('--optimizer', type=str, default='adam', help='adam or sgd')#优化器
parser.add_argument('--weight_init', type=str, default='kaiming', help='normal, xavier, kaiming, orhtogonal weights initialisation')#初始化权重
parser.add_argument('--weight_decay', type=float, default=0, help='L2 weight decay/regularisation on?')#L2权重劝退
parser.add_argument('--lr_decay', action='store_true', help='decay learning rate with rule')#学习率衰退
parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate')#起始学习率
parser.add_argument('--niter_decay', type=int, default=400, help='# of iter to linearly decay learning rate to zero')#学习率线性衰退
parser.add_argument('--lr_policy', type=str, default=None, help='{}learning rate policy: lambda|step|plateau')#学习率方法
parser.add_argument('--lr_decay_iters', type=int, default=7, help='multiply by a gamma every lr_decay_iters iterations')#学习率乘法
parser.add_argument('--clip_grad_norm', type=int, default=0, help='performs gradient clipping')#渐变剪裁
parser.add_argument('--gamma', type=float, default=0.5, help='factor to decay learning rate every lr_decay_iters with')

# Loss settings
parser.add_argument('--loss_criterion', type=str, default='mse', choices=allowed_losses(), help="loss criterion")#损失标准
parser.add_argument('--print_freq', type=int, default=10000, help="print every x iterations")#每X次迭代一次
parser.add_argument('--save_freq', type=int, default=100000, help="save every x interations")#每X迭代保存一次
parser.add_argument('--metric', type=str, default='rmse', choices=allowed_metrics(), help="metric to use during evaluation")#评估度量
parser.add_argument('--metric_1', type=str, default='mae', choices=allowed_metrics(), help="metric to use during evaluation")#评估度量
parser.add_argument('--wlid', type=float, default=0.1, help="weight base loss")#基础权重损失
parser.add_argument('--wrgb', type=float, default=0.1, help="weight base loss")
parser.add_argument('--wpred', type=float, default=1, help="weight base loss")
parser.add_argument('--wguide', type=float, default=0.1, help="weight base loss")
# Cudnn
parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True,
                    default=True, help="cudnn optimization active")#GPU使用
parser.add_argument('--gpu_ids', default='1', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')#GPU ID
parser.add_argument("--multi", type=str2bool, nargs='?', const=True,
                    default=False, help="use multiple gpus")#多个GPU
parser.add_argument("--seed", type=str2bool, nargs='?', const=True,
                    default=True, help="use seed")#使用种子
parser.add_argument("--use_disp", type=str2bool, nargs='?', const=True,
                    default=False, help="regress towards disparities")#？
parser.add_argument('--num_samples', default=500, type=int, help='number of samples')#样本数目
# distributed training
parser.add_argument('--world_size', default=1, type=int,
                    help='number of distributed processes')#分布式进程数
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--local_rank', dest="local_rank", default=0, type=int)


def main():
    global args#全局参数
    args = parser.parse_args()#读取参数
    if args.num_samples == 0:
        args.num_samples = None
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size
    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    channels_in = 4#输入通道读取
    model = Models.define_model(mod='New_net',max_depth=10)
    define_init_weights(model, args.weight_init)#网络权重初始化
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    if not args.no_cuda:
        if not args.multi:
            model = model.cuda()#单GPU
        else:
            model = torch.nn.DataParallel(model).cuda()#多GPU

    optimizer = define_optim(args.optimizer, model.parameters(), args.learning_rate, args.weight_decay)#优化器，权重参数，学习率，权重衰退
    criterion_local = define_loss(args.loss_criterion)
    criterion_lidar = define_loss(args.loss_criterion)
    criterion_rgb = define_loss(args.loss_criterion)
    criterion_guide = define_loss(args.loss_criterion)

    # INIT dataset
    dataset = Datasets.define_dataset(args.dataset, args.data_path, args.input_type, args.side_selection)
    dataset.prepare_dataset()
   
    print("Evaluate only")
    best_file_lst = glob.glob(os.path.join(args.save_path, 'model_best*'))
    if len(best_file_lst) != 0:
    	best_file_name = best_file_lst[0]
    	if os.path.isfile(best_file_name):
    		sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
    		print("=> loading checkpoint '{}'".format(best_file_name))
    		#name = './model_best_epoch_37.pth.tar'
    		checkpoint = torch.load(best_file_name)    		
    		model.load_state_dict(checkpoint['state_dict'])
    	else:
    		print("=> no checkpoint found at '{}'".format(best_file_name))
    else:
    	print("=> no checkpoint found at due to empy list in folder {}".format(args.save_path))
    score_valid, score_valid_1, losses_valid = validate(dataset, model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide) 
    print("=> Start validation set")
    print("===> Average RMSE score on validation set is {:.4f}".format(score_valid))
    print("===> Average MAE score on validation set is {:.4f}".format(score_valid_1))



def validate(dataset, model, criterion_lidar, criterion_rgb, criterion_local, criterion_guide, epoch=0):
    model.eval()
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    with torch.no_grad():
        for i, (img, rgb, gt) in tqdm(enumerate(zip(dataset.selected_paths['lidar_in'],
                                           dataset.selected_paths['img'], dataset.selected_paths['gt']))):

            raw_path = os.path.join(img)
            raw_pil = Image.open(raw_path)
            gt_path = os.path.join(gt)
            gt_pil = Image.open(gt)

            crop = 0
            raw_pil_crop = raw_pil.crop((0, crop, 640, 480))
            gt_pil_crop = gt_pil.crop((0, crop, 640, 480))

            raw = depth_read(raw_pil_crop, args.sparse_val)
            raw = to_tensor(raw).float()
            gt = depth_read(gt_pil_crop, args.sparse_val)
            gt = to_tensor(gt).float()
            valid_mask = (raw > 0).detach().float()

            input = torch.unsqueeze(raw, 0).cuda()
            gt = torch.unsqueeze(gt, 0).cuda()


            if args.input_type == 'rgb':
                rgb_path = os.path.join(rgb)
                rgb_pil = Image.open(rgb_path).convert('RGB')
                rgb_pil_crop = rgb_pil.crop((0, crop, 640, 480))
                rgb = to_tensor(rgb_pil_crop).float()
                rgb = torch.unsqueeze(rgb, 0).cuda()

                input = torch.cat((input, rgb), 1)

            torch.cuda.synchronize()
            output, rgbout = model(input)
            torch.cuda.synchronize()

            output = output * 6553.5
            raw = raw * 256.
            output = output[0][0:1].cpu()
            data = output[0].numpy()
    
            if crop != 0:
                padding = (0, 0, crop, 0)
                output = torch.nn.functional.pad(output, padding, "constant", 0)
                output[:, 0:crop] = output[:, crop].repeat(crop, 1)

            pil_img = to_pil(output.int())
            file_save_path = '/home/samidare/Results/nyu_new'
            if not os.path.exists(file_save_path):
            	os.mkdir(file_save_path)
            else:
            	pass
            pil_img.save(os.path.join(file_save_path, os.path.basename(img)))
    print('num imgs: ', i + 1)
    
    return 0, 0, 0
    
if __name__ == '__main__':
    main()








