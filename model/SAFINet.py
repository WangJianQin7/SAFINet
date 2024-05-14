import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models

__all__ = ["SAFINet"]

model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}

mob_conv1_2 = mob_conv2_2 = mob_conv3_3 = mob_conv4_3 = mob_conv5_3 = None


def conv_1_2_hook(module, input, output):
    global mob_conv1_2
    mob_conv1_2 = output
    return None


def conv_2_2_hook(module, input, output):
    global mob_conv2_2
    mob_conv2_2 = output
    return None


def conv_3_3_hook(module, input, output):
    global mob_conv3_3
    mob_conv3_3 = output
    return None


def conv_4_3_hook(module, input, output):
    global mob_conv4_3
    mob_conv4_3 = output
    return None


def conv_5_3_hook(module, input, output):
    global mob_conv5_3
    mob_conv5_3 = output
    return None


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.mbv = models.mobilenet_v2(pretrained=True).features

        self.mbv[1].register_forward_hook(conv_1_2_hook)
        self.mbv[3].register_forward_hook(conv_2_2_hook)
        self.mbv[6].register_forward_hook(conv_3_3_hook)
        self.mbv[13].register_forward_hook(conv_4_3_hook)
        self.mbv[17].register_forward_hook(conv_5_3_hook)

    def forward(self, x: Tensor) -> Tensor:
        global mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3
        self.mbv(x)

        return mob_conv1_2, mob_conv2_2, mob_conv3_3, mob_conv4_3, mob_conv5_3


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], 1)
        x2 = self.conv1(x1)

        return self.sigmoid(x2)


class SpatialAttention_no_sig(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention_no_sig, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], 1)
        x2 = self.conv1(x1)

        return x2


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias, stride=stride
    )


class MAF(nn.Module):
    def __init__(self, channel):
        super(MAF, self).__init__()
        self.branch0 = BasicConv2d(channel, channel, 1)

        self.branch1 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(channel, channel, 3, padding=3, dilation=3)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(channel, channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(channel, channel, 3, padding=5, dilation=5)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(channel, channel, 1),
            BasicConv2d(channel, channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channel, channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(channel, channel, 3, padding=7, dilation=7)
        )

        self.conv0 = nn.Conv2d(channel, 1, 1, bias=False)
        self.conv1 = nn.Conv2d(channel, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(channel, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(channel, 1, 1, bias=False)

        self.compress = BasicConv2d(channel * 4, channel, 3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_in):
        x0 = self.branch0(x_in)
        x_0 = self.sigmoid(self.conv0(x0)) * x0

        x1 = self.branch1(x_in + x0)
        x_1 = self.sigmoid(self.conv1(x1)) * x1

        x2 = self.branch2(x_in + x0 + x1)
        x_2 = self.sigmoid(self.conv2(x2)) * x2

        x3 = self.branch3(x_in + x0 + x1 + x2)
        x_3 = self.sigmoid(self.conv3(x3)) * x3

        x_maf = self.compress(torch.cat((x_0, x_1, x_2, x_3), 1)) + x_in

        return x_maf


class SCorrM(nn.Module):
    def __init__(self, channel=1):
        super(SCorrM, self).__init__()
        self.smooth = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.conv1 = nn.Conv2d(channel, channel, 1, bias=False)
        self.conv2 = nn.Conv2d(channel, channel, 1, bias=False)

    def forward(self, sa, fb_sa):
        fb_sa1 = F.interpolate(fb_sa, size=sa.size()[2:], mode='bilinear', align_corners=True)
        fb_sa1 = self.smooth(fb_sa1)

        W = sa * fb_sa1
        W_sa = F.softmax(W, dim=0)

        sa_1 = sa * W_sa
        fb_sa2 = fb_sa1 * W_sa

        sa_2 = self.conv1(sa_1 + fb_sa1)
        fb_sa3 = self.conv2(fb_sa2 + fb_sa1)

        A_scorr = sa_2 + fb_sa3

        return A_scorr


class FRAF(nn.Module):
    def __init__(self, channel, kernel_size=3, bias=False):
        super(FRAF, self).__init__()
        self.bconv = nn.Sequential(
            conv(channel, channel, kernel_size, bias=bias),
            nn.ReLU(inplace=True),
            conv(channel, channel, kernel_size, bias=bias)
        )

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()
        # self.sa1 = SpatialAttention()
        self.sa_no_sig = SpatialAttention_no_sig()

        self.SpatialCorrelation = SCorrM()

        self.FG_conv = BasicConv2d(channel, channel, 3, padding=1)
        self.BG_conv = BasicConv2d(channel, channel, 3, padding=1)
        self.FBG_conv = BasicConv2d(channel * 2, channel, 3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, fb_sa=None):
        x_conv = self.bconv(x)
        x_ca = self.ca(x_conv) * x_conv

        if fb_sa is not None:
            A_sa_st = self.sa_no_sig(x_ca)
            A_sa = self.sigmoid(self.SpatialCorrelation(A_sa_st, fb_sa))
        else:
            A_sa = self.sa(x_ca)

        x_fg = self.FG_conv(A_sa * x_ca)

        x_bg = self.BG_conv((1 - A_sa) * x_ca)

        x_fraf = self.FBG_conv(torch.cat((x_fg, x_bg), 1)) + x

        return x_fraf


class FRAF2(nn.Module):
    def __init__(self, channel, kernel_size=3, bias=False):
        super(FRAF2, self).__init__()
        self.bconv = nn.Sequential(
            conv(channel, channel, kernel_size, bias=bias),
            nn.ReLU(inplace=True),
            conv(channel, channel, kernel_size, bias=bias)
        )

        self.ca = ChannelAttention(channel)
        self.sa_no_sig = SpatialAttention_no_sig()

        self.FG_conv = BasicConv2d(channel, channel, 3, padding=1)
        self.BG_conv = BasicConv2d(channel, channel, 3, padding=1)
        self.FBG_conv = BasicConv2d(channel * 2, channel, 3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_conv = self.bconv(x)
        x_ca = self.ca(x_conv) * x_conv

        A_fbsa = self.sa_no_sig(x_ca)

        A_sa = self.sigmoid(A_fbsa)

        x_fg = self.FG_conv(A_sa * x_ca)

        x_bg = self.BG_conv((1 - A_sa) * x_ca)

        x_fraf = self.FBG_conv(torch.cat((x_fg, x_bg), 1)) + x

        return x_fraf, A_fbsa


class SAFINet(nn.Module):
    def __init__(self, channel=32):
        super(SAFINet, self).__init__()
        # Backbone model
        self.encoder = MobileNet()

        self.Translayer1 = Reduction(16, channel)
        self.Translayer2 = Reduction(24, channel)
        self.Translayer3 = Reduction(32, channel)
        self.Translayer4 = Reduction(96, channel)
        self.Translayer5 = Reduction(320, channel)

        self.maf1 = MAF(channel)
        self.maf2 = MAF(channel)
        self.maf3 = MAF(channel)
        self.maf4 = MAF(channel)
        self.maf5 = MAF(channel)

        self.fraf1 = FRAF(channel)
        self.fraf2 = FRAF2(channel)
        self.fraf3 = FRAF(channel)
        self.fraf4 = FRAF(channel)
        self.fraf5 = FRAF(channel)

        self.upsample21 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample22 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample23 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample24 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.compress1 = BasicConv2d(channel * 2, channel, 1)
        self.compress2 = BasicConv2d(channel * 2, channel, 1)
        self.compress3 = BasicConv2d(channel * 2, channel, 1)
        self.compress4 = BasicConv2d(channel * 2, channel, 1)

        self.s_conv1 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv2 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv3 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv4 = nn.Conv2d(channel, 1, 3, padding=1)
        self.s_conv5 = nn.Conv2d(channel, 1, 3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        size = x.size()[2:]
        f1, f2, f3, f4, f5 = self.encoder(x)

        s1 = self.Translayer1(f1)
        s2 = self.Translayer2(f2)
        s3 = self.Translayer3(f3)
        s4 = self.Translayer4(f4)
        s5 = self.Translayer5(f5)

        fb_sa = None
        x2_fraf = x3_fraf = x4_fraf = x5_fraf = None

        for cycle in range(3):
            x5_maf = self.maf5(s5)
            if fb_sa is not None:
                x5_fraf = self.fraf5(x5_maf, fb_sa)
            else:
                x5_fraf = self.fraf5(x5_maf)

            x4_1 = torch.cat((s4, self.upsample24(x5_fraf)), 1)
            x4_1 = self.compress4(x4_1)
            x4_maf = self.maf4(x4_1)
            if cycle > 0:
                x4_fraf = self.fraf4(x4_maf, fb_sa)
            else:
                x4_fraf = self.fraf4(x4_maf)

            x3_1 = torch.cat((s3, self.upsample23(x4_fraf)), 1)
            x3_1 = self.compress3(x3_1)
            x3_maf = self.maf3(x3_1)
            if cycle > 0:
                x3_fraf = self.fraf3(x3_maf, fb_sa)
            else:
                x3_fraf = self.fraf3(x3_maf)

            x2_1 = torch.cat((s2, self.upsample22(x3_fraf)), 1)
            x2_1 = self.compress2(x2_1)
            x2_maf = self.maf2(x2_1)
            x2_fraf, fb_sa = self.fraf2(x2_maf)

        x1_1 = torch.cat((s1, self.upsample21(x2_fraf)), 1)
        x1_1 = self.compress1(x1_1)
        x1_maf = self.maf1(x1_1)
        x1_fraf = self.fraf1(x1_maf)

        sal_out = self.s_conv1(x1_fraf)
        x2_out = self.s_conv2(x2_fraf)
        x3_out = self.s_conv3(x3_fraf)
        x4_out = self.s_conv4(x4_fraf)
        x5_out = self.s_conv5(x5_fraf)

        sal_out = F.interpolate(sal_out, size=size, mode='bilinear', align_corners=True)
        x2_out = F.interpolate(x2_out, size=size, mode='bilinear', align_corners=True)
        x3_out = F.interpolate(x3_out, size=size, mode='bilinear', align_corners=True)
        x4_out = F.interpolate(x4_out, size=size, mode='bilinear', align_corners=True)
        x5_out = F.interpolate(x5_out, size=size, mode='bilinear', align_corners=True)

        return sal_out, self.sigmoid(sal_out), x2_out, self.sigmoid(x2_out), x3_out, self.sigmoid(x3_out), x4_out, self.sigmoid(x4_out), x5_out, self.sigmoid(x5_out)
