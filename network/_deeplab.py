import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class CustomConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=2, mu=0.1, bias=False, *args, **kwargs):
        super(CustomConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)  # all pairs are considered to be of form (n, n)
        self.dilation = _pair(dilation)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.mu = mu
        self.fuzzy_weight = nn.Parameter(torch.Tensor(out_channels))

        self.calculated_kernel_size = (self.dilation[0] * (self.kernel_size[0] - 1) + 1, self.dilation[1] * (self.kernel_size[1] - 1) + 1)
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size[0], dilation=dilation, stride=stride, bias=bias, *args, **kwargs)
        self.fuzzy_conv = nn.Conv2d(self.in_channels, 1, self.calculated_kernel_size, stride=stride, bias=False, *args, **kwargs)
        shape = self.fuzzy_conv.weight.shape
        del self.fuzzy_conv.weight
        self.fuz = self.mask_dial(self.kernel_size[0], self.dilation[0], self.mu).broadcast_to(shape)
        with torch.no_grad():
            self.fuzzy_conv.weight = nn.Parameter(self.fuz, requires_grad=False)

        nn.init.kaiming_uniform_(self.conv.weight)

    def mask_dial(self, kernel_size, dilation, mu):
        dilation -= 1
        mid = [0 for i in range(dilation)]
        lim = (dilation // 2) if (dilation % 2 == 0) else ((dilation // 2) + 1)
        diff = (1 - mu) / lim
        filter1 = []
        for i in range(lim):
            mid[i] = 1 - (i + 1) * diff
            mid[dilation - 1 - i] = 1 - (i + 1) * diff
        for i in range(2 * kernel_size - 1):
            if i % 2 == 0:
                filter1 = filter1 + [0]
            else:
                filter1 = filter1 + mid
        filter2 = [[0 for i in range(dilation + 2)] for j in range(dilation)]
        for i in range(lim):
            for j in range(i + 2):
                filter2[i][j] = mid[i]
                filter2[i][dilation + 1 - j] = mid[i]
                filter2[dilation - i - 1][j] = mid[i]
                filter2[dilation - i - 1][dilation + 1 - j] = mid[i]
            for j in range(i + 1, lim):
                filter2[i][j + 1] = mid[j]
                filter2[i][dilation - j] = mid[j]
                filter2[dilation - i - 1][j + 1] = mid[j]
                filter2[dilation - i - 1][dilation - j] = mid[j]
        filter3 = [x[1:] for x in filter2]
        for i in range(kernel_size - 2):
            for j in range(len(filter2)):
                filter2[j] += filter3[j]
        result = []
        for i in range(2 * kernel_size - 1):
            if i % 2 == 0:
                result = result + [filter1]
            else:
                result = result + filter2
                result = [0 for i in range(2 * kernel_size - 1)]
        result = []
        for i in range(2 * kernel_size - 1):
            if i % 2 == 0:
                result = result + [filter1]
            else:
                result = result + filter2
        result = torch.Tensor(result)
        return result

    def forward(self, input_):
        if self.padding[0] > 0:
            padr = torch.zeros(input_.size()[0], input_.size()[1], input_.size()[2], self.padding[0])
            padc = torch.zeros(input_.size()[0], input_.size()[1], self.padding[1], input_.size()[3] + self.padding[0] * 2)
            input_ = torch.cat((input_, padr), 3)
            input_ = torch.cat((padr, input_), 3)
            input_ = torch.cat((input_, padc), 2)
            input_ = torch.cat((padc, input_), 2)

        conv_out = self.conv(input_)
        fuzzy_out = (self.fuzzy_conv(input_).permute(0, 2, 3, 1) * self.fuzzy_weight).permute(0, 3, 1, 2)

        return conv_out + fuzzy_out.broadcast_to(conv_out.shape)


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
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)

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
            CustomConv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

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
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            # nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            CustomConv(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
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
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
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
