import torch.nn as nn
from .blocks import (
    BlockB,
    ResNet,
    Model,
    VGGModel,
    Vgg16BN,
    ResNetFeatureExtractor,
    VGGFeatureExtractor,
    BidirectionalLSTM
)


class SeraQModel(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.stages = {
            "Trans": opt.Transformation,
            "Feat": opt.FeatureExtraction,
            "Seq": opt.SequenceModeling,
            "Pred": opt.Prediction
        }

        """ FeatureExtraction """
        if opt.FeatureExtraction == "VGG":
            self.FeatureExtraction = VGGFeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNetFeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception("No FeatureExtraction module specified")
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        if opt.SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print("No SequenceModeling module specified")
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == "CTC":
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def forward(self, inputs, text, is_train=True):
        """ Transformation stage """
        if not self.stages["Trans"] == "None":
            inputs = self.Transformation(inputs)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(inputs)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages["Seq"] == "BiLSTM":
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature

        """ Prediction stage """
        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(
                contextual_feature.contiguous()
            )
        else:
            prediction = self.Prediction(
                contextual_feature.contiguous(),
                text,
                is_train,
                batch_max_length=self.opt.batch_max_length
            )

        return prediction
