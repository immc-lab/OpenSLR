from models.base import Container , SignLanguageModel
from models.modules import ResNet, TemporalConv1D, BiLSTM, Classifier, Decoder, TLPLoss
from models.senmodules.senresnet import SENresnet
from models.senmodules.SENLoss import SENLoss
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
def build_sen(args, gloss_dict, loss_weights):
    return SignLanguageModel(
        spatial_module_container=Container([
            SENresnet(args)
        ]),
        temporal_module_container=Container([
            TemporalConv1D(args),
            BiLSTM(args),
            Classifier(args)
        ]),
        loss_module_container=Container([
            SENLoss(loss_weights)
        ]),
        decoder=Decoder(args,gloss_dict)
    )
def build_vac(args, gloss_dict, loss_weights):
    pass

def build_cvt(args, gloss_dict, loss_weights):
    pass

def build_slowfast(args, gloss_dict, loss_weights):
    pass

def build_corrnet(args, gloss_dict, loss_weights):
    pass
