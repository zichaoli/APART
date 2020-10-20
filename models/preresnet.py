import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['preact20']

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.info ={}
        self.layerwise = False
        self.alpha= torch.nn.Parameter(torch.FloatTensor([0]))

        

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def input_hook(self, grad):
        self.info['input_grad'] = grad.clone().detach()
    def input_hook2(self, grad):
        self.info['input_grad2'] = grad.detach().clone()

    def forward(self, x):


        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x

        if out.requires_grad == True:
            out.register_hook(self.input_hook)
        #Add noise to blocks
        if self.layerwise == True:
            grad = self.info['input_grad']
            noise = grad.sign() * 1
            delta = self.alpha * noise 
            out = out + delta
    

        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.alpha = torch.nn.Parameter(torch.FloatTensor([0.0001]))
        self.alpha_omega = torch.nn.Parameter(torch.FloatTensor([0.0001]))
        self.info = {}

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def input_hook(self, grad):
        self.info['input_grad'] = grad.clone().detach()

    def forward(self, x, layerwise=False):
        if x.requires_grad == True:
            x.register_hook(self.input_hook)        

        if layerwise == True:
            for block in self.layer1:
                block.layerwise = True
            for block in self.layer2:
                block.layerwise = True   
            for block in self.layer3:
                block.layerwise = True 
            for block in self.layer4:
                block.layerwise = True
        if layerwise == False:
            for block in self.layer1:
                block.layerwise = False
            for block in self.layer2:
                block.layerwise = False        
            for block in self.layer3:
                block.layerwise = False 
            for block in self.layer4:
                block.layerwise = False
        

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def preact20(num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2])
