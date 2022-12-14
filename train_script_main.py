'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
# from utils import progress_bar
from tqdm import tqdm
from xgen_tools import xgen_record, xgen_init, xgen_load, XgenArgs
from co_lib import Co_Lib as CL

COCOPIE_MAP = {'train_data_path': XgenArgs.cocopie_train_data_path,
               'eval_data_path': XgenArgs.cocopie_eval_data_path,
               'lr':'learning_rate'}

os.environ['MKL_THREADING_LAYER'] = 'GNU'

def training_main(args_ai=None):
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    args, args_ai = xgen_init(args, args_ai, COCOPIE_MAP)

    if 'CUDA_VISIBLE_DEVICES' in args_ai['general']:
        gpus = args_ai['general']['CUDA_VISIBLE_DEVICES']
        gpus = len(gpus.split(','))
        new_batch_size = args.batch_size * gpus
        args.batch_size = new_batch_size
        args_ai['origin']['batch_size'] = new_batch_size



    t_epoch = args_ai['origin']['common_train_epochs']

    scaling_factor = args_ai['origin']['scaling_factor']

    dummy_input = torch.rand(1, 3, 32, 32)

    num_workers = 4

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.train_data_path, train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=args.eval_data_path, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers)


    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    net_size =[2,2,2,2]
    net_size[0] = 2 if scaling_factor <6 else 3
    net_size[1] = 2 if scaling_factor < 6 else 4
    net_size[2] = scaling_factor
    net_size[3] = 2 if scaling_factor < 6 else 3
    net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=args.num_classes)
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()
    net = net.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    xgen_load(net, args_ai)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_epoch)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # Cocopie pruning 4: add prune_update_loss ***********************************************************************************************
            loss = CL.update_loss(loss)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f'train loss {train_loss}')
        print(f'training top-1 accuracy is {correct*100/total}')
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    def test(epoch, sim=None):
        global best_acc, dummy_input
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        # Save checkpoint.
        acc = 100. * correct / total
        print(f'testing top-1 accuracy is {acc}')
        return acc

    print('original:')
    test(-2)

    # Cocopie pruneing 1:add init function ***********************************************************************************************************
    # CL = CoLib()
    CL.init(args=args_ai, model=net, optimizer=optimizer, data_loader=trainloader)

    # mylogger = CL.logger
    print('after harden:')
    test(-2)

    best_acc = 0
    for epoch in range(start_epoch, start_epoch + t_epoch):
        # Cocopie pruning 2: add prune_update ********************************************************************************************************
        CL.before_each_train_epoch(epoch=epoch)
        train(epoch)
        acc = test(epoch)
        xgen_record(args_ai, net, acc, epoch=epoch)
        # mylogger.info(f"acc is {acc}")
        scheduler.step()
        # Cocopie pruning 3: add prune_update_learning_rate ******************************************************************************************
        CL.after_scheduler_step(epoch=epoch)

    acc = test(-1)
    xgen_record(args_ai, net, acc, epoch=-1)

    # return args_ai


if __name__ == '__main__':
    training_main()

