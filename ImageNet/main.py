# Additive Power-of-Two Quantization: An Efficient Non-uniform Discretization For Neural Networks
# Yuhang Li, Xin Dong, Wei Wang
# International Conference on Learning Representations (ICLR), 2020.


import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import models
from models.quant_layer import *
from tensorboardX import SummaryWriter
import sys
import gc

import wandb
import numpy as np


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names)
parser.add_argument('-j','--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--bit', default=4, type=int, help='the bit-width of the quantized network')
parser.add_argument('--data', metavar='DATA_PATH', default='~/shared/Imagenet_data/',
                    help='path to imagenet data (default: ./data/)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--percentage', default=-100, type=int,
                    help='bot std percentage of filters to quantize to po2 in each layer')

best_acc1 = 0


GPU_UUID_list = ["MIG-a2d6ade7-2495-5f5f-ba67-150f0038f7be", "MIG-975aecf0-aaec-5c79-a8bc-9b005ecd72dd",\
 "MIG-62de5551-207b-5aa3-8ce2-51d7a95276b6", "MIG-43b6e1cb-c7ab-5311-87f1-3a0abaa7ec82",\
 "MIG-2b9f5334-d823-56aa-85d3-83a0feef722b", "MIG-8c49a915-7cfb-5331-8e6c-77434700c634",\
 "MIG-b670fdd7-3c7e-5ee7-826d-f4fbbc4d6369",\
 \
 "MIG-fd11dc37-b7f0-5b58-8f6c-1f1c8ab246cf", "MIG-3a6ee76b-0551-511d-ac62-c222282db8ea",\
 "MIG-0cd2f85e-57fd-5d1c-849f-ffe0a84e10dd",\
 \
 "MIG-7c492f81-7d8e-5c27-8448-e8dd104e6dcb", "MIG-475a44c1-946f-51ba-a37e-c6f03a7d7eb0",\
 \
 "MIG-4d2d7c00-50b3-5e32-8e06-48e418e72ee8", "MIG-974c54a7-ee81-5fe6-999d-28227fff589a"]


# os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_UUID_list[0], GPU_UUID_list[1], GPU_UUID_list[2], GPU_UUID_list[3], GPU_UUID_list[4], GPU_UUID_list[5], GPU_UUID_list[6]}"
# os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-7116f551-f1c3-2265-eb56-c90cd826ee1a"







def main():
    args = parser.parse_args()

    # project_name = f"apot_{args.arch}_bw{args.bit}_lr{args.lr}_bz{args.batch_size}_epochs{args.epochs}"
    # wandb.init(project = project_name, entity = "piggybusuk",\
    # name=f"{args.arch}\
    # bw_{args.bit}_lr{args.lr}_batch{args.batch_size}"
    # )
    # wandb.config = {
    #     "learning_rate": args.lr,
    #     "batch_size": args.batch_size,
    #     "train_epochs": args.epochs,
    #     "Model_name": args.arch,
    # }






    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global device
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    # model = models.__dict__[args.arch](pretrained=True, bit=args.bit)

    ############ args로 입력해준 std BOT perc%를 argument로 넣어서 model load
    model = models.__dict__[args.arch](pretrained=True, bit=args.bit, percent=f"{args.percentage}")



    # for k, v in model.__dict__.items():
    #         if isinstance(v, torch.Tensor):
    #             v.cuda()





    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            # model = torch.nn.DataParallel(model, device_ids=[1]).cuda()
   

    # init from pre-trained model or full-precision model
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pre-trained model from {}".format(args.pretrained))
            print(f"cuda dev count: {torch.cuda.device_count()}")
            model = torch.nn.DataParallel(model, device_ids=[args.gpu]).cuda()
            checkpoint = torch.load(args.pretrained, map_location='cuda:0')
            model.load_state_dict(checkpoint['model'])
   

            # ############## apot 체크포인트 불러올때, module.module. key에러 뜨는 문제 해결

            # state_dict = torch.load(args.pretrained, map_location='cuda:0')['state_dict']
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:] # remove `module.`
            #     new_state_dict[name] = v

            # model.load_state_dict(new_state_dict)



            # ################### apot로 훈련되지 않아서, alpha가 없는 model을 불러올때
            # # model.load_state_dict(checkpoint['model_state_dict'])
            # # model.load_state_dict(checkpoint)

            # # print(f"model_state_dict keys: {checkpoint['model_state_dict'].keys()}")

            # # for key in list(checkpoint['model_state_dict'].keys()):
            # for key in list(checkpoint['state_dict'].keys()):
            #     if 'module.module.' not in key and 'module.' in key:
            #         # checkpoint['model_state_dict'][key.replace(f"{key}", 'module.'.join(key))] = checkpoint['model_state_dict'][key]
            #         # checkpoint['model_state_dict']['module.'.join(key)] = checkpoint['model_state_dict'][key]

            #         # checkpoint['model_state_dict'].update({ f"module.{key}" : checkpoint['model_state_dict'][key] } )
            #         # del checkpoint['model_state_dict'][key]

            #         checkpoint['state_dict'].update({ f"module.{key}" : checkpoint['state_dict'][key] } )
            #         del checkpoint['state_dict'][key]

            #     # if 'layer' in key and 'conv' in key:
            #     #     checkpoint['model_state_dict'].update({ f"module.{key.replace('weight', 'weight_alpha')}" : torch.nn.Parameter(torch.tensor(3.0)) } )
            #     #     checkpoint['model_state_dict'].update({ f"module.{key.replace('weight', 'act_alpha')}" : torch.nn.Parameter(torch.tensor(6.0)) } )
                
            #     # if 'layer' in key and 'downsample' in key:
            #     #     checkpoint['model_state_dict'].update({ f"module.{key.replace('weight', 'weight_alpha')}" : torch.nn.Parameter(torch.tensor(3.0)) } )
            #     #     checkpoint['model_state_dict'].update({ f"module.{key.replace('weight', 'act_alpha')}" : torch.nn.Parameter(torch.tensor(6.0)) } )
                

            
            # # print(f"New Keys: {checkpoint['model_state_dict'].keys()}")
            # # model.load_state_dict(checkpoint['model_state_dict'])
            # model.load_state_dict(checkpoint['state_dict'])


            model.cuda()
            model.module.show_params()
        else:
            print('no pre-trained model found')
            exit()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # customize the lr and wd for clipping thresholds
    # model_params = []
    # for name, params in model.module.named_parameters():
    #     if 'act_alpha' in name:
    #         model_params += [{'params': [params], 'lr': 3e-2, 'weight_decay': 2e-5}]
    #     elif 'wgt_alpha' in name:
    #         model_params += [{'params': [params], 'lr': 1e-2, 'weight_decay': 1e-4}]
    #     else:
    #         model_params += [{'params': [params]}]
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e+6))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
           
           
           ############ module.이 붙어서 저장된걸, 없는채로 load해야 하는 문제 해결
           

            for key in list(checkpoint['state_dict'].keys()):
                if  'module.' not in key:
                    checkpoint['state_dict'].update({ f"module.{key}" : checkpoint['state_dict'][key] } )
                    del checkpoint['state_dict'][key]

           
           
           
           
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True



    # ################# (맨 아래 주석처리한 부분에 관련된 파트이므로 무시) pot quantization


    # model.train()
    # perc_in=args.percentage
    # po2_quant(model, args.bit, args.percentage)
    # print(f"pot{args.bit}  {perc_in}% custom pt quantization finished!")

    # ################# (맨 아래 주석처리한 부분에 관련된 파트이므로 무시) 


    # data loader by official torchversion:
    # --------------------------------------------------------------------------
    print('==> Using Pytorch Dataset')
    input_size = 224  # image resolution for resnets
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    torchvision.set_image_backend('accimage')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    # --------------------------------------------------------------------------
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    writer = SummaryWriter(comment='res18_4bit')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)
        scheduler.step()
        writer.add_scalar('test_acc', acc1, epoch)
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        print('best_acc:' + str(best_acc1))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        # wandb.watch(model, None, log="all", log_freq=10)

        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # loss.backward() # 원본
        loss.backward(retain_graph=True) # ResNet에서 Mixed 했을때
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        gc.collect()
    
    # wandb.log({"epoch": epoch, "train_accuracy": top1.avg})
    
    
    writer.add_scalar('train_acc', top1.avg, epoch)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # wandb.log({"epoch": i, "test_accuracy": top1.avg})

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    # model.module.show_params()

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoints.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'res18_4best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adjust_list = [30,60,80,100]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()














# ################### (폐기, quant_layer.py 및 resnet.py에서 구현함) Filter Wise Quantization을 구현하는 블록 시작
    


# def sort_qscheme(tensor1, coef):  ### index는 tensor로 나타냄 (수정후)
       
#     th = coef * ( torch.max(torch.std(tensor1, (1, 2, 3))).item() - torch.min(torch.std(tensor1, (1, 2, 3))).item()) + torch.min(torch.std(tensor1, (1, 2, 3))).item()
#     qs1id = (torch.std(tensor1, (1, 2, 3)) > th).nonzero()
#     qs2id = (torch.std(tensor1, (1, 2, 3)) <=th).nonzero()

#     return qs1id , qs2id


# def sep_wt_mod(tensor1, coef):   # for문 없애고

#     # sort_qscheme에서 만든 index tensor 2개 쓰면 되지 않나?
#     b_ind1 = sort_qscheme(tensor1, coef)[0][:, 0]
#     b_ind2 = sort_qscheme(tensor1, coef)[1][:, 0]
     
#     tensor_qs1 = tensor1[b_ind1, :, :, :].to(device) 
#     tensor_qs2 = tensor1[b_ind2, :, :, :].to(device) 

#     return tensor_qs1, tensor_qs2




# def gradient_scale(x, scale):
#     yout = x
#     ygrad = x * scale
#     y = (yout - ygrad).detach() + ygrad
#     return y


# def uniform_quantization(tensor, alpha, bit, is_weight=True, grad_scale=None):
#     if grad_scale:
#         alpha = gradient_scale(alpha, grad_scale)
#     data = tensor / alpha
#     if is_weight:
#         data = data.clamp(-1, 1)
#         data = data * (2 ** (bit - 1) - 1)
#         data_q = (data.round() - data).detach() + data
#         data_q = data_q / (2 ** (bit - 1) - 1) * alpha
#     else:
#         data = data.clamp(0, 1)
#         data = data * (2 ** bit - 1)
#         data_q = (data.round() - data).detach() + data
#         data_q = data_q / (2 ** bit - 1) * alpha
#     return data_q



# # With Additive
# def build_power_value(B=2, additive=False):
#     base_a = [0.]
#     base_b = [0.]
#     base_c = [0.]
#     base_d = [0.] #내가 추가
#     if additive:
#         if B == 2:
#             for i in range(3):
#                 base_a.append(2 ** (-i - 1))
#         elif B == 4:
#             for i in range(3):
#                 base_a.append(2 ** (-2 * i - 1))
#                 base_b.append(2 ** (-2 * i - 2))
#         elif B == 6:
#             for i in range(3):
#                 base_a.append(2 ** (-3 * i - 1))
#                 base_b.append(2 ** (-3 * i - 2))
#                 base_c.append(2 ** (-3 * i - 3))
#         elif B == 3:
#             for i in range(3):
#                 if i < 2:
#                     base_a.append(2 ** (-i - 1))
#                 else:
#                     base_b.append(2 ** (-i - 1))
#                     base_a.append(2 ** (-i - 2))
#         elif B == 5:
#             for i in range(3):
#                 if i < 2:
#                     base_a.append(2 ** (-2 * i - 1))
#                     base_b.append(2 ** (-2 * i - 2))
#                 else:
#                     base_c.append(2 ** (-2 * i - 1))
#                     base_a.append(2 ** (-2 * i - 2))
#                     base_b.append(2 ** (-2 * i - 3))
        
#         elif B == 7: #내가 추가
#             for i in range(4):
#                 if i<3:
#                     base_a.append(2 ** (-3 * i - 1))
#                     base_b.append(2 ** (-3 * i - 2))
#                     base_c.append(2 ** (-3 * i - 3))
                
#                 else:
#                     base_d.append(2 ** (-3 * 1 - 1))
#                     base_a.append(2 ** (-3 * i - 2))
#                     base_b.append(2 ** (-3 * i - 3))
#                     base_c.append(2 ** (-3 * i - 4))

#         else:
#             pass
#     else:
#         for i in range(2 ** B - 1):
#             base_a.append(2 ** (-i - 1))
#     values = []
#     for a in base_a:
#         for b in base_b:
#             for c in base_c:
#                 values.append((a + b + c))
#     values = torch.Tensor(list(set(values))).to(device) 
#     values = values.mul(1.0 / torch.max(values))
#     return values


# # With Additive
# def pot_quantization(tensor, alpha, proj_set, is_weight=True, grad_scale=None):
#     def power_quant(x, value_s):
#         if is_weight:
#             shape = x.shape
#             xhard = x.view(-1)
#             sign = x.sign()
#             value_s = value_s.type_as(x)
#             xhard = xhard.abs()
#             idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
#             xhard = value_s[idxs].view(shape).mul(sign)
#             xhard = xhard
#         else:
#             shape = x.shape
#             xhard = x.view(-1)
#             value_s = value_s.type_as(x)
#             xhard = xhard
#             idxs = (xhard.unsqueeze(0) - value_s.unsqueeze(1)).abs().min(dim=0)[1]
#             xhard = value_s[idxs].view(shape)
#             xhard = xhard
#         xout = (xhard - x).detach() + x
#         return xout

#     if grad_scale:
#         alpha = gradient_scale(alpha, grad_scale)
#     data = tensor / alpha
#     if is_weight:
#         data = data.clamp(-1, 1)
#         data_q = power_quant(data, proj_set)
#         data_q = data_q * alpha
#     else:
#         data = data.clamp(0, 1)
#         data_q = power_quant(data, proj_set)
#         data_q = data_q * alpha
#     return data_q

# # With Additive
# def mixed_quant(tensor1, in_std_th, coef_alph, fixed_point_num_total_bits=8,  fixed_point_num_fraction_bits=4, po2_num_total_bits=8, in_additive=False): 

#     global acc_qs1 
#     global acc_qs2

#     w1=sep_wt_mod(tensor1, in_std_th)[0].to(device)
#     w2=sep_wt_mod(tensor1, in_std_th)[1].to(device)
#     otensor = torch.empty(tensor1.shape).to(device)

#     weight_alpha = torch.nn.Parameter(torch.mul(torch.max(torch.abs(tensor1)).to(device), coef_alph))
#     if (coef_alph==3.0):
#         weight_alpha = torch.nn.Parameter(torch.tensor(3.0).to(device))

#     proj_set = build_power_value(po2_num_total_bits-1, in_additive)

 
#     qd1 = w1 
#     qd2 = pot_quantization(w2, weight_alpha, proj_set, is_weight=True, grad_scale=None).to(device)

#     # 합치는 파트
#     bind1 = sort_qscheme(tensor1, in_std_th)[0][:, 0] 
#     bind2 = sort_qscheme(tensor1, in_std_th)[1][:, 0] 
#     # otensor[bind1, :, :, :] = qd1
#     otensor[bind1, :, :, :] = w1
#     otensor[bind2, :, :, :] = qd2

#     return otensor


# def po2_quant(in_model, po2_bits=4, perc=100):

#     coeff = perc/100
#     qs1_tot=0
#     qs2_tot=0
#     j0=0

#     print(f"STD Bot {perc}%,  begin\n")

#     for name, mod in in_model.named_modules():
            
#         if isinstance(mod, nn.Conv2d):
            
#             num_qs1 = torch.numel(sort_qscheme(mod.weight, coeff)[0])
#             num_qs2 = torch.numel(sort_qscheme(mod.weight, coeff)[1])

#             qs1_tot+=num_qs1
#             qs2_tot+=num_qs2
        
#             mod.weight = torch.nn.Parameter(mixed_quant(mod.weight, coeff, torch.nn.Parameter(torch.tensor(3.0).to(device)), 8, 4, po2_bits, False))


#             j0+=1

    
#     print(f"STD Bot {perc}%: POT {100*qs2_tot/(qs1_tot+qs2_tot)}%, FP32 {100*qs1_tot/(qs1_tot+qs2_tot)}%")
    


# ################### (폐기, quant_layer.py 및 resnet.py에서 구현함) Filter Wise Quantization을 구현하는 블록 끝

