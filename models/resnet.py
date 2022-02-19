'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=1, bias=True):
        super().__init__()
        self.depthwise_conv=nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.pointwise_conv=nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias= bias
        )
        
    def forward(self, x):
        out=self.depthwise_conv(x)
        out=self.pointwise_conv(out)
        return out




class Channel_attention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        sum_result = max_result + avg_result
        sum_result=self.se(F.relu(sum_result)) #[bs, c, 1, 1]
        output=self.sigmoid(sum_result)*x
        return output
        
class Spatial_attention(nn.Module):
    def __init__(self, in_channel, reduction = 4):
        super(Spatial_attention, self).__init__()
        self.relu = nn.ReLU()
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(in_channel,1,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(1,1,1,bias=False),
            
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        bs, c, h, w = x.shape
        att_map = self.reduce_dim(self.relu(x)) #[bs, 1, h, w]
        att_map = self.sigmoid(att_map) #[bs, 1, h, w]
        spatial_attention = att_map*x
        return spatial_attention


class MDAM(nn.Module):
    def __init__(self, in_channels, reduce_factor=16, increase_factor=4, residual=True):
        super(MDAM, self).__init__()
        self.residual = residual
        self.channel_attention = Channel_attention(channel=in_channels)
        self.spatial_attention = Spatial_attention(in_channels)
        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.normal_(m.weight, std=0.001)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
    def forward(self, x):
        ori = x
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        if self.residual:
          x = x + ori
        return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, deepwise=False):
        super(BasicBlock, self).__init__()
        
        if deepwise == True:
            self.conv1 = DepthwiseSeparableConvolution(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = DepthwiseSeparableConvolution(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
            
        self.bn2 = nn.BatchNorm2d(planes)
        self.attention = MDAM(in_channels = planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.dr = nn.Dropout2d(p=0.1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.attention(out)
        out = self.dr(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out





class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=7):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, deepwise=False)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, deepwise=False)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, deepwise=True)
        self.linear = nn.Sequential(
            nn.Linear(512*block.expansion, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024,7),
           
        )
        self.maxpool = nn.MaxPool2d((2,2))
        self.adp = nn.AdaptiveAvgPool2d((1,1))

    def _make_layer(self, block, planes, num_blocks, stride, deepwise=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, deepwise))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.adp(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 3, 2, 2])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 1, 32, 32))
    print(y.size())
# test()
