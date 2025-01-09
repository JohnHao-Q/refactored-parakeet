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
from datetime import datetime
from Loss.loss import define_loss, allowed_losses, New_MSE_loss, Smooth_loss, All_MSE_loss, MAE_loss
from Loss.benchmark_metrics import Metrics, allowed_metrics
from Datasets.dataloader import get_loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Utils.utils import str2bool, define_optim, define_scheduler, \
                        Logger, AverageMeter, first_run, mkdir_if_missing, \
                        define_init_weights, init_distributed_mode
                        
parser = argparse.ArgumentParser(description='KITTI Depth Completion Task')
parser.add_argument('--dataset', type=str, default='kitti', choices=Datasets.allowed_datasets(), help='dataset to work with')
parser.add_argument('--nepochs', type=int, default=60, help='Number of epochs for training')#训练次数
parser.add_argument('--thres', type=int, default=0, help='epoch for pretraining')#预训练次数
parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch number for training')#起始训练次数
parser.add_argument('--mod', type=str, default='New_net', choices=Models.allowed_models(), help='Model for use')#训练模型
parser.add_argument('--batch_size', type=int, default=16, help='batch size')#批量大小
parser.add_argument('--val_batch_size', default=None, help='batch size selection validation set')#验证批量大小
parser.add_argument('--learning_rate', metavar='lr', type=float, default=1e-4, help='learning rate')#学习率
parser.add_argument('--no_cuda', action='store_true', help='no gpu usage')#GPU使用情况

parser.add_argument('--evaluate', action='store_true', help='only evaluate')#仅仅评估用
parser.add_argument('--resume', type=str, default='', help='resume latest saved run')#恢复最新的运行
parser.add_argument('--nworkers', type=int, default=8, help='num of threads')#线程
parser.add_argument('--nworkers_val', type=int, default=0, help='num of threads')#验证线程
parser.add_argument('--no_dropout', action='store_true', help='no dropout in network')#无权重衰减
parser.add_argument('--subset', type=int, default=6000, help='Take subset of train set')#训练集子集
parser.add_argument('--input_type', type=str, default='rgb', choices=['depth','rgb'], help='use rgb for rgbdepth')#输入类型
parser.add_argument('--side_selection', type=str, default='image_03', help='train on one specific stereo camera')#相机特定选择
parser.add_argument('--no_tb', type=str2bool, nargs='?', const=True,
                    default=True, help="use mask_gt - mask_input as final mask for loss calculation")#结果选择误差输出
parser.add_argument('--test_mode', action='store_true', help='Do not use resume')#测试模型
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=False, help='use pretrained model')#使用预训练模型
parser.add_argument('--load_external_mod', type=str2bool, nargs='?', const=True, default=False, help='path to external mod')#其他模型路径

# Data augmentation settings
parser.add_argument('--crop_w', type=int, default=640, help='width of image after cropping')#图像宽
parser.add_argument('--crop_h', type=int, default=480, help='height of image after cropping')#图像高
parser.add_argument('--max_depth', type=float, default=10, help='maximum depth of LIDAR input')#最大深度
parser.add_argument('--sparse_val', type=float, default=0.0, help='value to endode sparsity with')#填充值
parser.add_argument("--rotate", type=str2bool, nargs='?', const=True, default=False, help="rotate image")#图像旋转？
parser.add_argument("--flip", type=str, default='hflip', help="flip image: vertical|horizontal")#水平或垂直
parser.add_argument("--rescale", type=str2bool, nargs='?', const=True,
                    default=False, help="Rescale values of sparse depth input randomly")#深度输入随机缩放
parser.add_argument("--normal", type=str2bool, nargs='?', const=True, default=False, help="normalize depth/rgb input")#归一化输入
parser.add_argument("--no_aug", type=str2bool, nargs='?', const=True, default=False, help="rotate image")#图像旋转？

# Paths settings
parser.add_argument('--save_path', default='Saved/', help='save path')#保存路径
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
parser.add_argument('--wprimary', type=float, default=1, help="weight base loss")#基础权重损失
parser.add_argument('--wsmooth', type=float, default=0.001, help="weight base loss")
parser.add_argument('--wrgb', type=float, default=0.001, help="weight base loss")
parser.add_argument('--wip', type=float, default=0.1, help="weight base loss")
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
parser.add_argument('--num_samples', default=1000, type=int, help='number of samples')#训练样本数目
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
        random.seed(args.seed)#随机数记忆
        torch.manual_seed(args.seed)
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    torch.backends.cudnn.benchmark = args.cudnn#CUDNN 加速
    
    model = Models.define_model(mod=args.mod, max_depth=10)
    define_init_weights(model, args.weight_init)#网络权重初始化
    torch.cuda.empty_cache()
    if not args.no_cuda:
        if not args.multi:
            model = model.cuda()#单GPU
        else:
            model = torch.nn.DataParallel(model).cuda()#多GPU
            
            
    save_id = '{}_{}_{}_{}_{}_batch{}_pretrain{}_wprimary{}_wip{}_wsmooth{}_wrgb{}_patience{}_num_samples{}_multi{}'.\
              format(args.mod, args.optimizer, args.loss_criterion,
                     args.learning_rate,
                     args.input_type, 
                     args.batch_size,
                     args.pretrained, args.wprimary, args.wip, args.wsmooth, args.wrgb, 
                     args.lr_decay_iters, args.num_samples, args.multi)#保存名称
    optimizer = define_optim(args.optimizer, model.parameters(), args.learning_rate, args.weight_decay)#优化器，权重参数，学习率，权重衰退
    scheduler = define_scheduler(optimizer, args)#学习率方法选择
     # Optional to use different losses#损失函数调用
    loss_smooth = Smooth_loss()
    loss_part = New_MSE_loss()
    loss_all = All_MSE_loss()
    loss_L1 = MAE_loss()
     
     # INIT dataset
    dataset = Datasets.define_dataset(args.dataset, args.data_path, args.input_type, args.side_selection)#初始化为Kitti_preprocessing类
    dataset.prepare_dataset()#继承父类函数，获取数据集路径
    train_loader, valid_loader, valid_selection_loader = get_loader(args, dataset)#制作读取数据集
    
    pretrained = torch.load('model_best_epoch_38.pth.tar')
    model.load_state_dict(pretrained['state_dict'])
     
    best_epoch = 0
    lowest_loss = np.inf
    args.save_path = os.path.join(args.save_path, save_id)
    mkdir_if_missing(args.save_path)
    log_file_name = 'log_train_start_0.txt'
    args.resume = first_run(args.save_path)
    if args.resume and not args.test_mode and not args.evaluate:
        path = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(int(args.resume)))
        if os.path.isfile(path):
            log_file_name = 'log_train_start_{}.txt'.format(args.resume)
            # stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(path)
            args.start_epoch = checkpoint['epoch']
            lowest_loss = checkpoint['loss']
            best_epoch = checkpoint['best epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            log_file_name = 'log_train_start_0.txt'
           # stdout
            sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            print("=> no checkpoint found at '{}'".format(path))
            
    else:
    	sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
            
    print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
    print("Init model: '{}'".format(args.mod))
    print("Number of parameters in model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in model.parameters())/1e6))


    # Start training
    for epoch in range(args.start_epoch, args.nepochs):
        print("\n => Start EPOCH {}".format(epoch + 1))
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(args.save_path)
        # Adjust learning rate

        # Define container objects
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        score_train = AverageMeter()
        score_train_1 = AverageMeter()
        metric_train = Metrics(max_depth=args.max_depth, disp=args.use_disp, normal=args.normal)

        # Train model for args.nepochs
        model.train()

        # compute timing
        end = time.time()

        # Load dataset
        for i, (input, gt) in tqdm(enumerate(train_loader)):

            # Time dataloader
            data_time.update(time.time() - end)
            # Put inputs on gpu if possible
            if not args.no_cuda:
                input, gt = input.cuda(), gt.cuda()#, ip_gt.cuda()
            prediction, rgb_prediction = model(input)
            
            loss1 = loss_part(prediction, gt)
            loss2 = loss_L1(prediction, gt)
            loss3 = loss_smooth(prediction, input[:, 1:, :, :])
            loss4 = loss_part(rgb_prediction, gt)
            loss = args.wprimary*loss1 + args.wip*loss2 + args.wsmooth*loss3+args.wrgb*loss4#0.8,0.3,0.1,0.3

            losses.update(loss.item(), input.size(0))
            metric_train.calculate(prediction.detach(), gt.detach())
            score_train.update(metric_train.get_metric(args.metric), metric_train.num)
            score_train_1.update(metric_train.get_metric(args.metric_1), metric_train.num)

            # Clip gradients (usefull for instabilities or mistakes in ground truth)
            if args.clip_grad_norm != 0:
                nn.utils.clip_grad_norm(model.parameters(), args.clip_grad_norm)

            # Setup backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Time trainig iteration
            batch_time.update(time.time() - end)
            end = time.time()

            # Print info
            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Metric {score.val:.4f} ({score.avg:.4f})'.format(
                       epoch+1, i+1, len(train_loader), batch_time=batch_time,
                       loss=losses,
                       score=score_train))


        print("===> Average RMSE score on training set is {:.4f}".format(score_train.avg))
        print("===> Average MAE score on training set is {:.4f}".format(score_train_1.avg))
        print("===> Last best score was RMSE of {:.4f} in epoch {}".format(lowest_loss,
                                                                           best_epoch))
        score_valid, score_valid_1, losses_valid = validate(valid_loader, model, loss_part, loss_L1, loss_smooth, epoch)
        print("===> Average RMSE score on validation set is {:.4f}".format(score_valid))
        print("===> Average MAE score on validation set is {:.4f}".format(score_valid_1))
        
        total_score=score_train.avg                                                                   
        # Adjust lr if loss plateaued
        if args.lr_policy == 'plateau':
            scheduler.step(total_score)
            lr = optimizer.param_groups[0]['lr']
            print('LR plateaued, hence is set to {}'.format(lr))
        
        if args.lr_policy is not None and args.lr_policy != 'plateau':
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr is set to {}'.format(lr))
        
        # File to keep latest epoch
        with open(os.path.join(args.save_path, 'first_run.txt'), 'w') as f:
            f.write(str(epoch))

        # Save model
        to_save = False
        if total_score < lowest_loss:

            to_save = True
            best_epoch = epoch+1
            lowest_loss = total_score
        
        save_checkpoint({
            'epoch': epoch + 1,
            'best epoch': best_epoch,
            'arch': args.mod,
            'state_dict': model.state_dict(),
            'loss': lowest_loss,
            'optimizer': optimizer.state_dict()}, to_save, epoch)
    if not args.no_tb:
        writer.close()


def validate(loader, model, loss_part, loss_L1, loss_smooth, epoch=0):
    # batch_time = AverageMeter()
    losses = AverageMeter()
    metric = Metrics(max_depth=args.max_depth, disp=args.use_disp, normal=args.normal)
    score = AverageMeter()
    score_1 = AverageMeter()
    # Evaluate model
    model.eval()
    # Only forward pass, hence no grads needed
    with torch.no_grad():
        # end = time.time()
        for i, (input, gt) in tqdm(enumerate(loader)):
            if not args.no_cuda:
                input, gt = input.cuda(non_blocking=True), gt.cuda(non_blocking=True)
            prediction, rgb_prediction = model(input)

            loss_1 = loss_part(prediction, gt)
            loss_2 = loss_L1(prediction, gt)
            loss_3 = loss_smooth(prediction, input[:, 1:, :, :])
            loss_4 = loss_part(rgb_prediction, gt)
            loss = args.wprimary*loss_1 + args.wip*loss_2 + args.wsmooth*loss_3+args.wrgb*loss_4
            losses.update(loss.item(), input.size(0))

            metric.calculate(prediction[:, 0:1], gt)
            score.update(metric.get_metric(args.metric), metric.num)
            score_1.update(metric.get_metric(args.metric_1), metric.num)

            if (i + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Metric {score.val:.4f} ({score.avg:.4f})'.format(
                       i+1, len(loader), loss=losses,
                       score=score))
    return score.avg, score_1.avg, losses.avg

def save_checkpoint(state, to_copy, epoch):
    with torch.no_grad():
        filepath = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
        torch.save(state, filepath)
        if to_copy:
            if epoch > 0:
                lst = glob.glob(os.path.join(args.save_path, 'model_best*'))
                if len(lst) != 0:
                    os.remove(lst[0])
            shutil.copyfile(filepath, os.path.join(args.save_path, 'model_best_epoch_{}.pth.tar'.format(epoch)))
            print("Best model copied")
        if epoch > 0:
            prev_checkpoint_filename = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch-1))
            if os.path.exists(prev_checkpoint_filename):
                os.remove(prev_checkpoint_filename)


if __name__ == '__main__':
    main()
















                        
