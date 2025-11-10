from models.base import Container , SignLanguageModel
from models.modules import ResNet, TemporalConv1D, BiLSTM, Classifier, Decoder, TLPLoss

from models.modules.slowfast.SlowFast import SlowFast
from models.modules.slowfast.TemporalSlowFastConv1D import TemporalSlowFastConv1D
from models.modules.slowfast.temporal_model import temporal_model
from models.modules.slowfast.slowfast_classifier import slowfast_classifier
from models.modules.slowfast.slowfast_loss import slowfast_loss
from models.modules.slowfast.Decoder import SlowFast_Decoder


import torch.nn as nn


def build_tlp(args, gloss_dict, loss_weights):
    return SignLanguageModel(
        spatial_module_container = Container([
            ResNet(args)
        ]),
        temporal_module_container = Container([
            TemporalConv1D(args),
            BiLSTM(args),
            Classifier ( args )
        ]),
        loss_module_container = Container([
            TLPLoss(loss_weights)
        ]),
        decoder = Decoder(args, gloss_dict)
    )
    return model

def build_vac(args, gloss_dict, loss_weights):
    pass

def build_cvt(args, gloss_dict, loss_weights):
    pass

def build_slowfast(args, gloss_dict, loss_weights):

    return SignLanguageModel(
        spatial_module_container=Container([
            SlowFast(args)
        ]),

        temporal_module_container=Container([
            TemporalSlowFastConv1D(args),
            temporal_model(args),
            # slowfast_classifier(args)
        ]),

        loss_module_container=Container([
            slowfast_loss(loss_weights)
        ]),

        decoder=SlowFast_Decoder(args, gloss_dict)
    )
    return model

def build_corrnet(args, gloss_dict, loss_weights):
    pass

def build_sen(args, gloss_dict, loss_weights):
    pass