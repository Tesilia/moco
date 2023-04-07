#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import builtins
import os
import random
import shutil
import time
import warnings

import moco.loader
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torchvision.datasets import STL10
import torchvision.models as models
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from datetime import datetime


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet18", choices=model_names, help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",)
parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)",)
parser.add_argument("-b", "--batch-size", default=256, type=int, metavar="N", help="mini-batch size (default: 256), this is the total ""batch size of all GPUs on the current node when ""using Data Parallel or Distributed Data Parallel",)
parser.add_argument("--lr", "--learning-rate", default=30.0, type=float, metavar="LR", help="initial learning rate", dest="lr",)
parser.add_argument("--schedule", default=[60, 80], nargs="*", type=int, help="learning rate schedule (when to drop lr by a ratio)",)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--wd", "--weight-decay", default=0.0, type=float, metavar="W", help="weight decay (default: 0.)", dest="weight_decay",)
parser.add_argument("-p", "--print-freq", default=20, type=int, metavar="N", help="print frequency (default: 10)",)
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set",)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--results_dir", default='', type=str, metavar='PATH', help='path to cache')
parser.add_argument("--pretrained", default="", type=str, help="path to moco pretrained checkpoint")

best_acc1 = 0


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

    s = size + 32 
    if args.results_dir == '':
        try:
            os.mkdir('./saves/lincl/' + dataset)
        except:
            pass
        args.results_dir = './saves/lincl/' + dataset + '/' + datetime.now().strftime("%Y-%m-%d-%H-%M") + '/'
        try:
            os.mkdir(args.results_dir)
        except: 
            pass
    global best_acc1

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # load from pre-trained
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")
            
            if 'state_dict' in checkpoint: 
                checkpoint = checkpoint['state_dict']
            # rename moco pre-trained keys
            for k in list(checkpoint.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith("encoder_q") and not k.startswith("encoder_q.fc"):
                    # remove prefix
                    checkpoint[k[len("encoder_q.") :]] = checkpoint[k]
                # delete renamed or unused k
                del checkpoint[k]
            args.start_epoch = 0
            msg = model.load_state_dict(checkpoint, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        #print(name, param)
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False
    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    if torch.cuda.is_available():
        model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias
    optimizer = torch.optim.SGD(
        parameters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, "train")
    testdir = os.path.join(args.data, "test") if 'tiny' not in args.data else os.path.join(args.data, "val")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    augmentation_train = [
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose(augmentation_train))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=16,
        shuffle = True, 
        pin_memory=True,
    )

    augmentation_test = [
        transforms.Resize(s),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ]

    test_dataset = datasets.ImageFolder(testdir, transforms.Compose(augmentation_test))
    
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    
    bn_splits = ''
    if 'bn_splits' in args.pretrained:
        bn_splits = args.pretrained[-16]

    prefix = "{}_{}_lr_{}_bn_splits_{}".format(dataset, args.batch_size, args.lr, bn_splits)
    
    writer = SummaryWriter(logdir=args.results_dir)
    global_count = 0
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)
        
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args, writer, global_count)
        global_count += 1
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if epoch == args.epochs-1:
            print("Best Top1 Accuracy:{:2.2f}".format(best_acc1))
        save_checkpoint(
            args,
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            filename= "{}.pth".format(prefix),
        )
        if epoch == args.start_epoch:
            sanity_check(model.state_dict(), args.pretrained)
    writer.close()

def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss 35.94 ( 41.73)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, writer, global_count):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

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
        

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )
    writer.add_scalar('loss', float(losses.getAvg()), global_count)
    writer.add_scalar('acc_1', float(top1.getAvgAcc()), global_count)
    writer.add_scalar('acc_5', float(top5.getAvgAcc()), global_count)
    return top1.avg


def save_checkpoint(args, state, is_best, filename="checkpoint.pth"):
    
    if is_best:
        torch.save(state, os.path.join(args.results_dir, filename))


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint["state_dict"]

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if "fc.weight" in k or "fc.bias" in k:
            continue

        # name in pretrained model
        k_pre = (
            "encoder_q." + k[len("") :]
            if k.startswith("")
            else "encoder_q." + k
        )

        assert (
            state_dict[k].cpu() == state_dict_pre[k_pre]
        ).all(), "{} is changed in linear classifier training.".format(k)

    print("=> sanity check passed.")


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
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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
