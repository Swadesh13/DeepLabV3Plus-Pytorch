import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
import numpy as np
from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class FuzzyConv(nn.Module):
    """
    Fuzzy Conv Layer
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, mu=0.1, bias=False, *args, **kwargs):
        super(FuzzyConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)  # all pairs are considered to be of form (n, n)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.mu = mu

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias, *args, **kwargs)
        self.fuzzy_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=(self.dilation[0] + 1, self.dilation[1] + 1), padding="same", groups=in_channels, bias=False
        )
        self.fuzzy_weights = self.fuzzy_filter()
        with torch.no_grad():
            assert (
                self.fuzzy_conv.weight.shape == self.fuzzy_weights.shape
            ), f"Weight shape not matching, {self.fuzzy_weights.shape} vs {self.fuzzy_conv.weight.shape}"
            self.fuzzy_conv.weight = nn.Parameter(self.fuzzy_weights, requires_grad=False)
            self.fuzzy_conv.requires_grad_(False)

    def fuzzy_filter(self):
        fh, fw = self.dilation[0] + 1, self.dilation[1] + 1
        filter = np.zeros((fh, fw))
        for e, v in enumerate(np.linspace(self.mu, 1, int(self.dilation[0] / 2) + 1)):
            for i in range(int(self.dilation[0] / 2)):
                filter[e : fh - e, e : fw - e] = v

        filter = np.array(np.broadcast_to(filter, self.fuzzy_conv.weight.shape))

        return torch.Tensor(filter)

    def forward(self, x):
        x = self.fuzzy_conv(x)
        x = self.conv(x)
        return x


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36], stride_rates=[1, 1, 1]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate, stride_rates)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature["low_level"])
        output_feature = self.aspp(feature["out"])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode="bilinear", align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature["out"])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """Atrous Separable Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            # nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            FuzzyConv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        # self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, stride):
        modules = [
            # nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            FuzzyConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, stride_rates=[1, 1, 1]):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        stride1, stride2, stride3 = tuple(stride_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1, stride1))
        modules.append(ASPPConv(in_channels, out_channels, rate2, stride2))
        modules.append(ASPPConv(in_channels, out_channels, rate3, stride3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(
            module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.bias
        )
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module
