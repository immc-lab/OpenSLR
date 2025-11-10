from torch.nn.functional import conv1d

from models.base import Container , SignLanguageModel
from models.modules import ResNet, TemporalConv1D, BiLSTM, Classifier, Decoder, TLPLoss, VACLoss, VACTemporalConv1D
from models.modules.norm import NormLinear


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
    conv1d = VACTemporalConv1D ( args )
    classifier = Classifier( args )
    classifier.classifier = NormLinear(1024, args["num_classes"])
    conv1d.conv1d.fc = classifier.classifier
    return SignLanguageModel (
        spatial_module_container = Container ( [
            ResNet (args)
        ] ) ,
        temporal_module_container = Container ( [
            conv1d,
            BiLSTM ( args ) ,
            classifier
        ] ) ,
        loss_module_container = Container ( [
            VACLoss ( loss_weights )
        ] ) ,
        decoder = Decoder ( args , gloss_dict )
    )
    return model

def build_cvt(args, gloss_dict, loss_weights):
    pass

def build_slowfast(args, gloss_dict, loss_weights):
    pass

def build_corrnet(args, gloss_dict, loss_weights):
    pass

def build_sen(args, gloss_dict, loss_weights):
    pass