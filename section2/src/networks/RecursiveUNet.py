# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Defines the Unet.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1 at the bottleneck

# recursive implementation of Unet
import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, num_classes=3, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=4, norm_layer=nn.InstanceNorm3d):
        super(UNet, self).__init__()

        unet_block = UnetSkipConnectionBlock(
            in_channels=initial_filter_size * 2 ** (num_downs - 1),
            out_channels=initial_filter_size * 2 ** num_downs,
            num_classes=num_classes,
            kernel_size=kernel_size,
            norm_layer=norm_layer,
            innermost=True
        )
        for i in range(1, num_downs):
            unet_block = UnetSkipConnectionBlock(
                in_channels=initial_filter_size * 2 ** (num_downs - (i + 1)),
                out_channels=initial_filter_size * 2 ** (num_downs - i),
                num_classes=num_classes,
                kernel_size=kernel_size,
                submodule=unet_block,
                norm_layer=norm_layer
            )
        unet_block = UnetSkipConnectionBlock(
            in_channels=in_channels,
            out_channels=initial_filter_size,
            num_classes=num_classes,
            kernel_size=kernel_size,
            submodule=unet_block,
            norm_layer=norm_layer,
            outermost=True
        )

        self.model = unet_block

    def forward(self, x):
        return self.model(x)

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, num_classes=1, kernel_size=3,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        pool = nn.MaxPool3d(2, stride=2)
        conv1 = self.contract(in_channels, out_channels, kernel_size, norm_layer)
        conv2 = self.contract(out_channels, out_channels, kernel_size, norm_layer)
        conv3 = self.expand(out_channels * 2, out_channels, kernel_size)
        conv4 = self.expand(out_channels, out_channels, kernel_size)

        if outermost:
            final = nn.Conv3d(out_channels, num_classes, kernel_size=1)
            down = [conv1, conv2]
            up = [conv3, conv4, final]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(in_channels * 2, in_channels, kernel_size=2, stride=2)
            model = [pool, conv1, conv2, upconv]
        else:
            upconv = nn.ConvTranspose3d(in_channels * 2, in_channels, kernel_size=2, stride=2)
            down = [pool, conv1, conv2]
            up = [conv3, conv4, upconv]
            model = down + [submodule] + up + ([nn.Dropout(0.5)] if use_dropout else [])

        self.model = nn.Sequential(*model)

    @staticmethod
    def contract(in_channels, out_channels, kernel_size=3, norm_layer=nn.InstanceNorm3d):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            norm_layer(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    @staticmethod
    def expand(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    @staticmethod
    def center_crop(layer, target_d, target_h, target_w):
        b, c, d, h, w = layer.size()
        d1 = (d - target_d) // 2
        h1 = (h - target_h) // 2
        w1 = (w - target_w) // 2
        return layer[:, :, d1:(d1 + target_d), h1:(h1 + target_h), w1:(w1 + target_w)]

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            out = self.model(x)
            crop = self.center_crop(out, x.size(2), x.size(3), x.size(4))
            return torch.cat([x, crop], 1)
