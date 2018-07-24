from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch
#from torch_deform_conv.layers import ConvOffset2D
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, FCN=False, radius=1., thresh=0.5):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.FCN=FCN

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet.__factory[depth](pretrained=pretrained)

#==========================add dilation=============================#
        if self.FCN:
            for mo in self.base.layer4[0].modules():
                if isinstance(mo, nn.Conv2d):
                    mo.stride = (1,1)
#================append conv for FCN==============================#
            self.num_features = num_features
            self.num_classes = 751 #num_classes
            self.dropout = dropout
            out_planes = self.base.fc.in_features
            self.local_conv = nn.Conv2d(out_planes, self.num_features, kernel_size=1,padding=0,bias=False)
            init.kaiming_normal(self.local_conv.weight, mode= 'fan_out')
#            init.constant(self.local_conv.bias,0)
            self.feat_bn2d = nn.BatchNorm2d(self.num_features) #may not be used, not working on caffe
            init.constant(self.feat_bn2d.weight,1) #initialize BN, may not be used
            init.constant(self.feat_bn2d.bias,0) # iniitialize BN, may not be used

##---------------------------stripe1----------------------------------------------#
            self.instance0 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance0.weight, std=0.001)
            init.constant(self.instance0.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance1 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance1.weight, std=0.001)
            init.constant(self.instance1.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance2 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance2.weight, std=0.001)
            init.constant(self.instance2.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance3 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance3.weight, std=0.001)
            init.constant(self.instance3.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance4 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance4.weight, std=0.001)
            init.constant(self.instance4.bias, 0)
##---------------------------stripe1----------------------------------------------#
##---------------------------stripe1----------------------------------------------#
            self.instance5 = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.instance5.weight, std=0.001)
            init.constant(self.instance5.bias, 0)

            self.drop = nn.Dropout(self.dropout)

        elif not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            self.radius = nn.Parameter(torch.FloatTensor([radius]))
            self.thresh = nn.Parameter(torch.FloatTensor([thresh]))



            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features, bias=False)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes,bias=True)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x
#=======================FCN===============================#
        if self.FCN:
            y = x.unsqueeze(1)
            y = F.avg_pool3d(x,(16,1,1)).squeeze(1)
            sx = x.size(2)/6
            kx = x.size(2)-sx*5
            x = F.avg_pool2d(x,kernel_size=(kx,x.size(3)),stride=(sx,x.size(3)))   # H4 W8
#========================================================================#            

            out0 = x.view(x.size(0),-1)
            out0 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            x = self.drop(x)
            x = self.local_conv(x)
            out1 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
            x = self.feat_bn2d(x)
            x = F.relu(x) # relu for local_conv feature
            
            x = x.chunk(6,2)
            x0 = x[0].contiguous().view(x[0].size(0),-1)
            x1 = x[1].contiguous().view(x[1].size(0),-1)
            x2 = x[2].contiguous().view(x[2].size(0),-1)
            x3 = x[3].contiguous().view(x[3].size(0),-1)
            x4 = x[4].contiguous().view(x[4].size(0),-1)
            x5 = x[5].contiguous().view(x[5].size(0),-1)
            
            c0 = self.instance0(x0)
            c1 = self.instance1(x1)
            c2 = self.instance2(x2)
            c3 = self.instance3(x3)
            c4 = self.instance4(x4)
            c5 = self.instance5(x5)
            return out0, (c0, c1, c2, c3, c4, c5)

#==========================================================#


        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)
        out1 = x.view(x.size(0),-1)
        center = out1.mean(0).unsqueeze(0).expand_as(out1)
        out2 = x/x.norm(2,1).unsqueeze(1).expand_as(x)
        
        if self.has_embedding:
            x = self.feat(x)
            out3 = x/ x.norm(2,1).unsqueeze(1).expand_as(x)
            x = self.feat_bn(x)
        
        if self.norm:
            x = x / x.norm(2, 1).unsqueeze(1).expand_as(x)
        elif self.has_embedding:  # adding relu after fc, not used in softmax but in tripletloss
            x = F.relu(x)
            out4 = x/ x.norm(2,1).unsqueeze(1).expand_as(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        return out2, x, out2, out2
	

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
