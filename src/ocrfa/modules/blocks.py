import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from torchvision import models
from collections import namedtuple
from packaging import version


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class Vgg16BN(nn.Module):
    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        if version.parse(torchvision.__version__) >= version.parse("0.13"):
            vgg_pretrained_features = models.vgg16_bn(
                weights=models.VGG16_BN_Weights.DEFAULT if pretrained else None
            ).features
        else:
            models.vgg.model_urls["vgg16_bn"] = models.vgg.model_urls[  # type:ignore
                "vgg16_bn"
            ].replace("https://", "http://")
            vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 19):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(19, 29):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(29, 39):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())  # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

    def forward(self, input_state):
        h = self.slice1(input_state)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"])
        out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
        return out


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inputs):
        """
        input_state : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        try:
            self.rnn.flatten_parameters()
        except:
            ...
        output = self.linear(self.rnn(inputs)[0])
        return output


class VGGFeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=256):
        super().__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel
        ]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(
                input_channel,
                self.output_channel[0],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                self.output_channel[0],
                self.output_channel[1],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                self.output_channel[1],
                self.output_channel[2],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
            nn.Conv2d(
                self.output_channel[2],
                self.output_channel[2],
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(
                self.output_channel[2],
                self.output_channel[3],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.Conv2d(
                self.output_channel[3],
                self.output_channel[3],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(
                self.output_channel[3],
                self.output_channel[3],
                kernel_size=2,
                stride=1,
                padding=0
            ),
            nn.ReLU(True)
        )

    def forward(self, input_state):
        return self.ConvNet(input_state)


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=512):
        super().__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BlockB, [1, 2, 5, 3])

    def forward(self, input_state):
        return self.ConvNet(input_state)


class BlockB(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, down_sample=None):
        super().__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = down_sample
        self.stride = stride

    @staticmethod
    def _conv3x3(in_planes, out_planes, stride=1):
        return nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + residual)


class ResNet(nn.Module):

    def __init__(self, input_channel, output_channel, block, layers):
        super().__init__()

        self.output_channel_block = [
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
            output_channel
        ]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(
            input_channel,
            int(output_channel / 16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(
            int(output_channel / 16),
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(
            self.output_channel_block[0],
            self.output_channel_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.max_pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.layer2 = self._make_layer(
            block,
            self.output_channel_block[1],
            layers[1],
            stride=1
        )
        self.conv2 = nn.Conv2d(
            self.output_channel_block[1],
            self.output_channel_block[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, self.output_channel_block[2], layers[2], stride=1)
        self.conv3 = nn.Conv2d(
            self.output_channel_block[2],
            self.output_channel_block[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(block, self.output_channel_block[3], layers[3], stride=1)
        self.conv4_1 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=(2, 1),
            padding=(0, 1),
            bias=False
        )
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            down_sample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, down_sample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(
            self.bn0_2(
                self.conv0_2(
                    self.relu(
                        self.bn0_1(
                            self.conv0_1(x)
                        )
                    )
                )
            )
        )
        x = self.relu(
            self.bn1(
                self.conv1(
                    self.layer1(
                        self.max_pool1(x)
                    )
                )
            )
        )
        x = self.relu(
            self.bn2(
                self.conv2(
                    self.layer2(
                        self.max_pool2(x)
                    )
                )
            )
        )
        x = self.relu(
            self.bn3(
                self.conv3(
                    self.layer3(
                        self.max_pool3(x)
                    )
                )
            )
        )
        return self.relu(
            self.bn4_2(
                self.conv4_2(
                    self.relu(
                        self.bn4_1(
                            self.conv4_1(
                                self.layer4(x)
                            )
                        )
                    )
                )
            )
        )


class VGGModel(nn.Module):
    def __init__(
            self,
            input_channel,
            output_channel,
            hidden_size,
            num_class
    ):
        super().__init__()

        self.FeatureExtraction = VGGFeatureExtractor(input_channel, output_channel)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.Prediction = nn.Linear(hidden_size, num_class)

    def forward(self, inputs, *args, **kwargs):
        return self.Prediction(
            self.SequenceModeling(
                self.AdaptiveAvgPool(
                    self.FeatureExtraction(inputs).permute(0, 3, 1, 2)
                ).squeeze(3)
            ).contiguous()
        )


class Model(nn.Module):

    def __init__(
            self,
            input_channel,
            output_channel,
            hidden_size,
            num_class
    ):
        super().__init__()
        self.FeatureExtraction = ResNetFeatureExtractor(input_channel, output_channel)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.Prediction = nn.Linear(hidden_size, num_class)

    def forward(self, inputs, *args, **kwargs):
        return self.Prediction(
            self.SequenceModeling(
                self.AdaptiveAvgPool(
                    self.FeatureExtraction(inputs).permute(0, 3, 1, 2)
                ).squeeze(3)
            ).contiguous()
        )
