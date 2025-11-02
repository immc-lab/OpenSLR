import pdb
import copy
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base import Container , SignLanguageModel
from models.modules.criterions import SeqKD
from models.modules import BiLSTMLayer , TemporalConv, Identity


class ResNet( nn.Module ) :
    def __init__ ( self, args) :
        super ( ResNet , self ).__init__ ( )
        c2d_type = args["c2d_type"]
        self.conv2d = getattr ( models , c2d_type ) ( weights = "ResNet18_Weights.IMAGENET1K_V1" )
        self.conv2d.fc = Identity ()

    def masked_bn ( self , inputs , len_x ) :
        def pad ( tensor , length ) :
            return torch.cat (
                [ tensor , tensor.new ( length - tensor.size ( 0 ) , *tensor.size ( ) [ 1 : ] ).zero_ ( ) ] )

        x = torch.cat ( [ inputs [ len_x [ 0 ] * idx :len_x [ 0 ] * idx + lgt ] for idx , lgt in enumerate ( len_x ) ] )
        x = self.conv2d ( x )
        x = torch.cat ( [ pad ( x [ sum ( len_x [ :idx ] ) :sum ( len_x [ :idx + 1 ] ) ] , len_x [ 0 ] )
                          for idx , lgt in enumerate ( len_x ) ] )
        return x

    def forward(self , data) :
        x = data['vid']
        len_x = data['vid_lgt']
        if len ( x.shape ) == 5 :
            # videos
            batch , temp , channel , height , width = x.shape
            inputs = x.reshape ( batch * temp , channel , height , width )
            framewise = self.masked_bn ( inputs , len_x )
            framewise = framewise.reshape ( batch , temp , -1 ).transpose ( 1 , 2 )
        else :
            # frame-wise features
            framewise = x
        return {
            "framewise_features": framewise,
            "visual_features" : x,
        }

class TemporalConv1D ( nn.Module ) :
    def __init__ ( self , args ) :
        super ( TemporalConv1D , self ).__init__ ( )
        self.conv1d = TemporalConv (
            input_size = 512 ,
            hidden_size = 1024 ,
            conv_type = args["conv_type"] ,
            use_bn = args["use_bn"] ,
            num_classes = args["num_classes"] ,
        )

    def forward ( self , data) :
        return self.conv1d ( data['framewise_features'] , data['vid_lgt'] )

class BiLSTM( nn.Module ) :
    def __init__ ( self , args) :
        super ( BiLSTM , self ).__init__ ( )
        self.temporal_model = BiLSTMLayer (
            rnn_type = 'LSTM' ,
            input_size = 1024 ,
            hidden_size = 1024 ,
            num_layers = 2 ,
            bidirectional = True
        )

    def forward ( self , data) :
        return self.temporal_model ( data [ 'visual_feat' ] , data [ 'feat_len' ] )

class Classifier ( nn.Module ) :
    def __init__ ( self , args ) :
        super ( Classifier , self ).__init__ ( )
        self.classifier = nn.Linear ( 1024 , args["num_classes"] )

    def forward ( self , data) :
        return {
            "sequence_logits": self.classifier ( data['predictions'] )
        }

class TLPLoss( nn.Module ) :
    def __init__ ( self, loss_weights) :
        super ( TLPLoss , self ).__init__ ( )
        self.loss = {}
        self.loss_weights = loss_weights
        self.loss [ 'CTCLoss' ] = torch.nn.CTCLoss ( reduction = 'none' , zero_infinity = False )
        self.loss [ 'distillation' ] = SeqKD ( T = 8 )

    def forward ( self , data ) :
        loss = 0
        total_loss = {}
        for k , weight in self.loss_weights.items ( ) :
            if k == 'ConvCTC' :
                total_loss [ 'ConvCTC' ] = weight * self.loss [ 'CTCLoss' ] (
                    data [ "conv_logits" ].log_softmax ( -1 ) ,
                    data["label"].cpu ( ).int ( ) , data [ "feat_len" ].cpu ( ).int ( ) ,
                    data["label_lgt"].cpu ( ).int ( ) ).mean ( )
                loss += total_loss [ 'ConvCTC' ]
            elif k == 'SeqCTC' :
                total_loss [ 'SeqCTC' ] = weight * self.loss [ 'CTCLoss' ] (
                    data [ "sequence_logits" ].log_softmax ( -1 ) ,
                    data["label"].cpu ( ).int ( ) , data [ "feat_len" ].cpu ( ).int ( ) ,
                    data["label_lgt"].cpu ( ).int ( ) ).mean ( )
                loss += total_loss [ 'SeqCTC' ]
            elif k == 'Dist' :
                total_loss [ 'Dist' ] = weight * self.loss [ 'distillation' ] ( data [ "conv_logits" ] ,
                                                                                data [
                                                                                    "sequence_logits" ].detach ( ) ,
                                                                                use_blank = False )
                loss += total_loss [ 'Dist' ]
            elif k == 'Cu' :
                total_loss [ 'Cu' ] = weight * data [ "loss_LiftPool_u" ]
                loss += total_loss [ 'Cu' ]
            elif k == 'Cp' :
                total_loss [ 'Cp' ] = weight * data [ "loss_LiftPool_p" ]
                loss += total_loss [ 'Cp' ]
        return {
            "loss": loss,
            "total_loss": total_loss
        }

class Decoder:
    def __init__ ( self , args, gloss_dict) :
        super ( Decoder , self ).__init__ ( )
        self.decoder = utils.Decode ( gloss_dict , args["num_classes"] , 'beam' )
    def __call__ ( self , data) :
        pred = self.decoder.decode ( data["sequence_logits"] , data["feat_len"] , batch_first = False , probs = False )
        return {
            "recognized_sents": pred
        }


def build_tlp(args, gloss_dict, loss_weights):
    resnet = ResNet(args)
    cnn1d = TemporalConv1D(args)
    lstm = BiLSTM(args)
    cls = Classifier(args)
    decoder = Decoder (args, gloss_dict)
    loss = TLPLoss(loss_weights)
    SpatioContainer = Container([resnet])
    TemporalContainer = Container([cnn1d, lstm, cls])
    LossContainer = Container([loss])
    model = SignLanguageModel(SpatioContainer, TemporalContainer, LossContainer, decoder)
    return model, cnn1d.conv1d.kernel_size