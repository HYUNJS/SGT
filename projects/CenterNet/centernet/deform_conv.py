import torch
import torch.nn as nn

from detectron2.layers import DeformConv, ModulatedDeformConv


class DLADeformConv(ModulatedDeformConv):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1):

        super(DLADeformConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, deformable_groups)
        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          deformable_groups * 3 * kernel_size[0] * kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        nn.init.constant_(self.conv_offset_mask.weight, 0)
        nn.init.constant_(self.conv_offset_mask.bias, 0)

    def forward(self, input):
        offset_mask = self.conv_offset_mask(input)
        offset_x, offset_y, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((offset_x, offset_y), dim=1)
        mask = mask.sigmoid()
        return super(DLADeformConv, self).forward(input, offset, mask)


class DeformConvWithOff(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1):
        super(DeformConvWithOff, self).__init__()
        self.offset_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcn = DeformConv(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            deformable_groups=deformable_groups,
        )

    def forward(self, input):
        offset = self.offset_conv(input)
        output = self.dcn(input, offset)
        return output


class ModulatedDeformConvWithOff(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1,
                 dilation=1, deformable_groups=1):
        super(ModulatedDeformConvWithOff, self).__init__()
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            deformable_groups * 3 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.dcnv2 = ModulatedDeformConv(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            deformable_groups=deformable_groups,
        )

    def forward(self, input):
        x = self.offset_mask_conv(input)
        o1, o2, mask = torch.chunk(x, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        output = self.dcnv2(input, offset, mask)
        return output
