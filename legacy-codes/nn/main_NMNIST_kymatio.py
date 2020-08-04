
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
# import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import os
import argparse

# from models.covnet import ResNet18
from utils import progress_bar, apply_rotations, print_in_logfile, AverageMeter, adjust_learning_rate
from models.mlp import EMLP

parser = argparse.ArgumentParser(description='PyTorch N-MNIST Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--model', default='resnet18', type=str, help='model to train (resnet18, revnet38)')
parser.add_argument('--resume-net', default='', type=str, metavar='PATH', help='path to the network checkpoint to resume (default: none)')
parser.add_argument('--logfile',  default='N-MNIST.log', type=str, help='filename')
parser.add_argument('--epochs-net', default=120, type=int, metavar='N', help='number of total epochs to run network training')
parser.add_argument('--n-layer-for-classification', default=3, type=int, metavar='N', help='number of the layer to use for classification')
parser.add_argument('--rotations', nargs='+', type=int, help='rotations (in 0, 90, 180, 270)', default=[0, 90, 180, 270])
args = parser.parse_args()

net  = EMLP(c_in= 4, c_out=[32,32],kernel_size=1000)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


test_best_acc = 0  # best test accuracy of the classifier on cifar
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

trainfeat = torch.tensor(np.load("/home/exarchakis/Datasets/N-MNIST2/train_feat.npy").astype(np.float32))
trainfeat[:,3,1:]=torch.from_numpy(np.diff(trainfeat[:,3],axis=1))
trainfeat[trainfeat<0]=0
trainlabs = torch.tensor(np.load("/home/exarchakis/Datasets/N-MNIST2/train_labels.npy").astype(np.int))
trainset = data.TensorDataset(trainfeat,trainlabs)
trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testfeat = torch.tensor(np.load("/home/exarchakis/Datasets/N-MNIST2/test_feat.npy").astype(np.float32))
testfeat[:,3,1:]=torch.from_numpy(np.diff(testfeat[:,3],axis=1))
testfeat[testfeat<0]=0
testlabs = torch.tensor(np.load("/home/exarchakis/Datasets/N-MNIST2/test_labels.npy").astype(np.int))
testset = data.TensorDataset(testfeat,testlabs)
testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

if args.resume_net:
    # Load checkpoint.
    print('==> Resuming network from checkpoint..')
    assert os.path.isfile(args.resume_net), 'Error: no checkpoint file found!'
    net_checkpoint = torch.load(args.resume_net)
    net.load_state_dict(net_checkpoint['net'])
    net_best_acc = net_checkpoint['acc']
    start_epoch = net_checkpoint['epoch']

net.to(device)

criterion = nn.CrossEntropyLoss()

# lr and optimizer for the self supervised task
lr = args.lr
logfile = args.logfile
optimizer = optim.Adam(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

def train(epoch):
    global lr, optimizer


    print('\nEpoch: %d' % epoch)

    train_losses = AverageMeter()

    adjust_learning_rate(optimizer, epoch, lr, rate=0.2, adjust_frequency=30)

    net.train()

    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # free stored activations for revnets
        if args.model in ['revnet18', 'revnet38']:
            net.module.free()

        optimizer.step()

        train_losses.update(loss.item(), inputs.size(0))
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f (%.3f) | Acc: %.3f%% (%d/%d)'
            % (train_losses.val, train_losses.avg , 100.*correct/total, correct, total))

    # import ipdb;ipdb.set_trace()
    print_in_logfile('Epoch %d, Train, Loss: %.3f, Acc: %.3f' % (epoch, train_losses.avg , 100.*correct/total), logfile)

def test(epoch):
    global test_best_acc
    net.eval()
    test_losses = AverageMeter()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)#, return_layer_number=args.n_layer_for_classification)

            loss = criterion(outputs, targets)

            test_losses.update(loss.item(), inputs.size(0))
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f (%.3f) | Acc: %.3f%% (%d/%d)'
                % (test_losses.val, test_losses.avg, 100.*correct/total, correct, total))
        print_in_logfile('Epoch %d, Test,  Loss: %.3f, Acc: %.3f' % (epoch, test_losses.avg, 100.*correct/total), logfile)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > test_best_acc:
        print('Saving..')
        state = {
			'net': net.state_dict(),
		    'acc': acc,
		    'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/network_ckpt.t7')
        test_best_acc = acc

print(trainfeat[0:20].shape)
print(net(trainfeat[0:20]).shape)
# import ipdb;ipdb.set_trace()
n_epochs_net = 110
print_in_logfile('Training network self-supervised for {} epochs'.format(args.epochs_net), logfile)
if args.resume_net:
    print_in_logfile('Network resumed from checkpoint {} (epoch {})'.format(args.resume_net, start_epoch), logfile)
for epoch in range(start_epoch, args.epochs_net):
    train(epoch)
    test(epoch)
