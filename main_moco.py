#!/usr/bin/env python

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import math
import os
import json
import random
import shutil
import time
import warnings
from datetime import datetime
import moco.builder
import moco.loader
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
from torchvision.datasets import STL10
import torchvision.models as models
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter



model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

# lr: 0.06 for batch 512 (or 0.03 for batch 256 or 0.015 for batch 128)
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet18", help="model architecture: " + " | ".join(model_names) + " (default: resnet50)")
parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=128, type=int, metavar="N", help="mini-batch size (default: 256)")
parser.add_argument("--lr","--learning-rate", default=0.015, type=float, metavar="LR", help="initial learning rate", dest="lr")
parser.add_argument("--schedule", default=[120, 160], nargs="*", type=int, help="learning rate schedule (when to drop lr by 10x)")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver")
parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)")
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--results-dir", default ='', type=str, metavar='PATH', help='path to cache (default: none')

# moco specific configs:
parser.add_argument("--moco-dim", default=128, type=int, help="feature dimension (default: 128)")
parser.add_argument("--moco-k", default=4096, type=int, help="queue size; number of negative keys (default: 65536)")
parser.add_argument("--moco-m", default=0.99, type=float, help="moco momentum of updating key encoder (default: 0.999)")
parser.add_argument("--moco-t", default=0.1, type=float, help="softmax temperature (default: 0.07)")
parser.add_argument("--bn-splits", default=4, type=int, help="simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu")
parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")

def main():
    args = parser.parse_args()

    dataset = ''
    size = 0
    if 'CIFAR' in args.data:
        dataset = 'cifar10'
        size = 32
    elif 'cub' in args.data:
        dataset = 'cub'
    elif 'miniImageNet' in args.data:
        dataset = 'miniimagenet'
        size = 224
    elif 'Omniglot' in args.data:
        dataset = 'omniglot'
    elif 'stl' in args.data:
        dataset = 'stl10'
        size = 96
    elif 'tiny' in args.data:
        dataset = 'tinyimagenet'
        size = 64

    if args.results_dir == '':
        try:
            os.mkdir('./saves/' + dataset)
        except:
            pass
        args.results_dir = './saves/' + dataset + '/' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '/'
        os.mkdir(args.results_dir)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
        arch=args.arch,
        bn_splits= args.bn_splits,
    )
    # print(model)
    model.cuda()
       
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    
    # Data loading code
    traindir = os.path.join(args.data,"train")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
    augmentation = [
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    # data prepare
    train_data = datasets.ImageFolder(traindir, moco.loader.TwoCropsTransform(transforms.Compose(augmentation))) if 'stl' not in args.data else STL10(root='/data/aponik/Data/stl10', split='train+unlabeled', transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)) , download=True)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True,
    )

    prefix = "{}_{}_lr_{}_bn_splits_{}".format(dataset, args.batch_size, args.lr, args.bn_splits)
    
    writer = SummaryWriter(logdir=args.results_dir)
    global_count = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer, global_count)
        global_count += 1
        if (epoch+1) == args.epochs:
            save_checkpoint(args, 
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename="{}_epochs_{:3d}.pth".format(prefix, epoch),
            )
    writer.close()

# Define train (for one epoch)
def train(train_loader, model, criterion, optimizer,  epoch, args, writer, global_count):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":2.4f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs),
    )

    # switch to train mode
    model.train()
    adjust_learning_rate(optimizer, epoch, args)

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)
        
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i == len(train_loader)-1:
            progress.display(i+1)

    writer.add_scalar('loss', float(losses.getAvg()), global_count)
    writer.add_scalar('acc_1', float(top1.getAvgAcc()), global_count)
    writer.add_scalar('acc_5', float(top5.getAvgAcc()), global_count)

def save_checkpoint(args, state, is_best, filename="checkpoint.pth"):
    torch.save(state, os.path.join(args.results_dir, filename))
    if is_best:
        shutil.copyfile(filename, "model_best.pth")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    
    def getAvg(self):
        return float('%.4f' %(self.avg))

    def getAvgAcc(self):
        return float('%.2f' %(self.avg))

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# precision_at_k bei moco lightening
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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    main()
