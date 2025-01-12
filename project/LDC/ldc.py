# Lightweight Dense CNN for Edge Detection
# It has less than 1 Million parameters

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from typing import Tuple

import pdb

# Below comes from https://github.com/jocpae/clDice
def soft_erode(img):
    p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
    p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
    return torch.min(p1,p2)

def soft_dilate(img):
    return F.max_pool2d(img, (3,3), (1,1), (1,1))

def soft_open(img):
    return soft_dilate(soft_erode(img))


def soft_skel(img, n: int):
    img1 = soft_open(img)
    skel =  F.relu(img - img1)
    for j in range(n):
        img  =  soft_erode(img)
        img1  =  soft_open(img)
        delta  =  F.relu(img - img1)
        skel  =  skel +  F.relu(delta - skel*delta)
    return skel


def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0,)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CoFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3,
                               stride=1, padding=1) # before 64
        self.conv3= nn.Conv2d(32, out_ch, kernel_size=3,
                               stride=1, padding=1)# before 64  instead of 32
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32) # before 64

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # to support torch.jit.script
        # x1, x2 = x
        # new_features = super(_DenseLayer, self).forward(F.relu(x1))  # F.relu()
        # return 0.5 * (new_features + x2), x2

        x1, x2 = x
        n = F.relu(x1)
        n = self.conv1(n)
        n = self.norm1(n)
        n = self.relu1(n)
        n = self.conv2(n)
        n = self.norm2(n)
        return 0.5 * (n + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor])->Tuple[torch.Tensor, torch.Tensor]:
        # to support torch.jit.script
        for layer in self:
            x = layer(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)

class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride,
                 use_bs=True
                 ):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)
        return x

class LDC(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self):
        super(LDC, self).__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2,)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(2, 32, 64) # [128,256,100,100]
        self.dblock_4 = _DenseBlock(3, 64, 96)# 128
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.side_2 = SingleConvBlock(32, 64, 2)

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(32, 64, 2)
        self.pre_dense_3 = SingleConvBlock(32, 64, 1)
        self.pre_dense_4 = SingleConvBlock(64, 96, 1)# 128

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(64, 2)
        self.up_block_4 = UpConvBlock(96, 3)# 128
        self.block_cat = CoFusion(4,4)# cats fusion method

        self.bgr_normal = T.Normalize(mean=[103.939, 116.779, 123.68 ], std=[1.0, 1.0, 1.0])

        self.apply(weight_init)

        self.load_weights()

        # pdb.set_trace()
        # torch.jit.script(self)

    def forward(self, x):
        B, C, H, W = x.size()
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=True)

        # Convert RGB[0.0, 1.0] to BGR[0, 255] !!!
        x = torch.cat([x[:, 2:3, :, :], x[:, 1:2, :, :], x[:, 0:1, :, :]], dim = 1) * 255.0
        x = self.bgr_normal(x)

        # x.size() -- [1, 3, 512, 512]

        # Block 1
        block_1 = self.block_1(x) # [8,16,176,176]
        block_1_side = self.side_1(block_1) # 16 [8,32,88,88]

        # Block 2
        block_2 = self.block_2(block_1) # 32 # [8,32,176,176]
        block_2_down = self.maxpool(block_2) # [8,32,88,88]
        block_2_add = block_2_down + block_1_side # [8,32,88,88]
        block_2_side = self.side_2(block_2_add) # [8,64,44,44] block 3 R connection

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down) # [8,64,88,88] block 3 L connection
        block_3, _ = self.dblock_3((block_2_add, block_3_pre_dense)) # [8,64,88,88]
        block_3_down = self.maxpool(block_3) # [8,64,44,44]
        block_3_add = block_3_down + block_2_side # [8,64,44,44]

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down) # [8,64,44,44]
        block_4_pre_dense = self.pre_dense_4(block_3_down+block_2_resize_half) # [8,96,44,44]
        block_4, _ = self.dblock_4((block_3_add, block_4_pre_dense)) # [8,96,44,44]


        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        # results = [out_1, out_2, out_3, out_4]

        # concatenate multiscale outputs
        block_cat = torch.cat([out_1, out_2, out_3, out_4], dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        out_1 = torch.sigmoid(out_1)
        out_2 = torch.sigmoid(out_2)
        out_3 = torch.sigmoid(out_3)
        out_4 = torch.sigmoid(out_4)
        block_cat = torch.sigmoid(block_cat)

        # output = torch.cat([out_1, out_2, out_3, out_4, block_cat], dim = 1)
        # output = output.mean(dim=1, keepdim=True)
        output = torch.sigmoid(block_cat)

        # Convert BGR to RGB !!!
        # output is Bx1xHxW, so no need convert channels
        output = (output - output.min())/(output.max() - output.min() + 1e-5)
        output = soft_skel(output, 3)
        output = (output - output.min())/(output.max() - output.min() + 1e-5)

        # kernel_size = 3
        # kernel = output.new_ones((1, 1, kernel_size, kernel_size))/(kernel_size * kernel_size)
        # output = F.conv2d(1.0 - output, kernel, stride=1, padding=kernel_size//2)
        # output = output.clamp(0.0, 0.9)
        # output = F.conv2d(output, kernel, stride=1, padding=kernel_size//2)
        # output = output.clamp(0.0, 0.9)
        # output = F.conv2d(output, kernel, stride=1, padding=kernel_size//2)
        # output = output.clamp(0.0, 0.9)
        # output = F.conv2d(output, kernel, stride=1, padding=kernel_size//2)
        # output = 1.0 - output

        output = F.interpolate(output, size=(H, W), mode="bilinear", align_corners=True)
        # output = (output >= 0.25).to(torch.float32)

        return output


    def load_weights(self, model_path="models/LDC.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading weight from {checkpoint} ...")
            self.load_state_dict(torch.load(checkpoint))
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"Weight file '{checkpoint}' not exist !!!")
