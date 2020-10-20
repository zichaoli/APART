"""
This file trains ATTA models on CIFAR10 dataset.
"""
from __future__ import print_function
import os
import sys
import argparse
from apex import amp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from models.preresnet import * 
from models.preresnet import PreActBlock
from adaptive_data_aug import atta_aug, atta_aug_trans, inverse_atta_aug
from attack import LinfPGDAttack
import json
import numpy as np



import cifar_dataloader



parser = argparse.ArgumentParser(description='PyTorch CIFAR TRADES Adversarial Training')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test_batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight_decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031, type=float,
                    help='perturbation')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_freq', '-s', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--eval_freq', '-e', default=2, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--gpuid', type=int, default=0,
                    help='The ID of GPU.')
parser.add_argument('--save_dir', type=str, default='./logs/')
parser.add_argument('--max_lr', default=0.2, type=float)
parser.add_argument('--max_lr_alpha', default=5e-8, type=float)
parser.add_argument('--weight_decay_alpha', default=400, type=float)
parser.add_argument('--layerwise', action='store_true', default=False)
parser.add_argument('--half', action='store_true', default=False,
                    help='use mixed precision')



args = parser.parse_args()


#Config file will overlap commend line args

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

# settings
save_dir = args.save_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")

# setup data loader

def train(args, model, device, cifar_nat_x, omega_sets, cifar_y, optimizer, optimizer_alpha, epoch, scheduler, scheduler_alpha):
    model.train()
    global writer
    num_of_example = 50000
    batch_size = args.batch_size
    cur_order = np.random.permutation(num_of_example)
    iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
    batch_idx = -batch_size
    total = 0
    correct = 0
    print(batch_size)
    for i in range(iter_num):
        batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
        omega = omega_sets[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]
        x_nat_batch = cifar_nat_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]
        y_batch = cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]]

        batch_size = y_batch.shape[0]

        #data augmentation
        rst = torch.zeros(batch_size,3,32,32).to(device)
        omega, transform_info = atta_aug(omega, rst)
        rst = torch.zeros(batch_size,3,32,32).to(device)
        x_nat_batch = atta_aug_trans(x_nat_batch, transform_info, rst)
        criterion_ce = nn.CrossEntropyLoss()



        model.train()
        optimizer.zero_grad()
        optimizer_alpha.zero_grad()
        #Initializatoin
        delta = model.alpha_omega * omega 
        delta.clamp_(-args.epsilon, args.epsilon)
        inputs = x_nat_batch.detach() + delta 
        inputs.clamp_(0,1)
        outputs = model(inputs)
        loss = criterion_ce(outputs, y_batch)
        if args.half:
            with amp.scale_loss(loss, [optimizer]) as scaled_loss:
                scaled_loss.backward(retain_graph=True)
        else:
            loss.backward(retain_graph=True)

        cur_grad = model.info['input_grad'].sign().detach()
        delta += (model.alpha) * (cur_grad)
        delta.clamp_(-args.epsilon, args.epsilon)
        adv_inputs = x_nat_batch.detach() + delta
        adv_inputs.clamp_(0,1)

        #Add layerwise noise
        outputs = model(adv_inputs, layerwise=args.layerwise)
        loss = criterion_ce(outputs, y_batch)
        optimizer.zero_grad()
        if args.half:
            with amp.scale_loss(loss, [optimizer]) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        omega = omega + (model.alpha.data.item() / (model.alpha_omega.data.item() + 1e-7)) * (model.info['input_grad'].sign().detach())
        omega.clamp_(-1, 1)

        #gradient ascent
        model.alpha.grad = - model.alpha.grad
        model.alpha_omega.grad = - model.alpha_omega.grad

        for p in model.modules():
            if isinstance(p, PreActBlock):
                p.alpha.grad = -p.alpha.grad


        optimizer_alpha.step()
        optimizer_alpha.zero_grad()
        optimizer.step()
        scheduler.step()
        scheduler_alpha.step()

        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, num_of_example,
                       100. * batch_idx / num_of_example, loss.item()))

        omega_sets[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = inverse_atta_aug(
            omega_sets[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]],
            omega, transform_info)
    train_acc = correct / total
    writer.add_scalar('train_acc', train_acc, global_step=epoch)
    print(" train acc: {:6.3f}%".format(100.0 * correct / total))



def test(args, model, device, epoch, test_loader):
    global writer

    model.eval()
    attack = LinfPGDAttack(epsilon=8/255, k=20, alpha=2/255)

    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()


    print(" === Validate PGD-20===")

    for batch_index, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        adv_inputs = attack(model, inputs, targets)
        outputs = model(adv_inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("   == adv loss: {:.3f} | adsv acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('adv_loss', test_loss, global_step=epoch)
    writer.add_scalar('adv_acc', test_acc, global_step=epoch)
    # Save checkpoint.
    acc = 100. * correct / total



def test_clean(args, model, device, epoch, test_loader):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    global writer
# set up data loader
    
    model.eval()
    attack = LinfPGDAttack(epsilon=8/255, k=20, alpha=2/255)

    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()


    print(" === Validate===")
    for batch_index, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print("   == clean loss: {:.3f} | clean acc: {:6.3f}%".format(
        test_loss / (batch_index + 1), 100.0 * correct / total))
    test_loss = test_loss / (batch_index + 1)
    test_acc = correct / total
    writer.add_scalar('test_loss', test_loss, global_step=epoch)
    writer.add_scalar('test_acc', test_acc, global_step=epoch)


def main():

    global writer
    writer = SummaryWriter(log_dir=args.save_dir +'/event')
    model = preact20().to(device)
    

    #Prepare data
    cifar_nat_x, cifar_y= cifar_dataloader.load_pading_training_data(device)
    omega_sets = torch.zeros_like(cifar_nat_x) + 0.001 * torch.randn(cifar_nat_x.shape).cuda().detach()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform_test = transforms.Compose([transforms.ToTensor(),])
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


    #Set optimziers and schedulers
    all_params = list(model.parameters())
    params = []
    for i in all_params:
        if len(i.data) == 1:
            params.append(i)
    optimizer = optim.SGD(model.parameters(), lr=args.max_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_alpha = optim.SGD(params, lr=args.max_lr_alpha, momentum=args.momentum, weight_decay=400)
    
    if args.half:
        model, [optimizer] = amp.initialize(model, [optimizer], opt_level="O1")
    
    num_of_example = 50000
    iter_num = num_of_example // args.batch_size + (0 if num_of_example % args.batch_size == 0 else 1)
    lr_steps = args.epochs * iter_num
    lr_steps_alpha = args.epochs * iter_num
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=0.2,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    scheduler_alpha = torch.optim.lr_scheduler.CyclicLR(optimizer_alpha, base_lr=0, max_lr=5e-8,
            step_size_up=lr_steps_alpha / 2, step_size_down=lr_steps_alpha / 2)


    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, cifar_nat_x, omega_sets, cifar_y, optimizer, optimizer_alpha, epoch, scheduler, scheduler_alpha)

        # evaluation 
        print('================================================================')
        if epoch % args.eval_freq == 0:
            # test(args, model, device)
            test_clean(args, model, device, epoch, test_loader)
            test(args, model, device, epoch, test_loader)
    writer.close()
    torch.save(model.state_dict(),
            os.path.join(save_dir, 'model-cifar-epoch{}.pt'.format(epoch)))

if __name__ == '__main__':
    main()