import torch.nn as nn

class Classifier ( nn.Module ) :
    def __init__ ( self , args ) :
        super ( Classifier , self ).__init__ ( )
        self.classifier = nn.Linear ( 1024 , args["num_classes"] )

    def forward ( self , data) :
        return {
            "sequence_logits": self.classifier ( data['predictions'] )
        }