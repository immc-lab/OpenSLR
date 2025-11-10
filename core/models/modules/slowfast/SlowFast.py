# modules/SlowFast.py
import torch
import torch.nn as nn
import torchvision.models as models
from models.modules.slowfast.slowfast_modules import slowfast

class SlowFast(nn.Module):
    def __init__(self, args):
        super(SlowFast, self).__init__()

        c2d_type = args.get("c2d_type", None)
        conv_type = args.get("conv_type", None)
        slowfast_config = args.get("slowfast_config", None)
        slowfast_args = args.get("slowfast_args", None)
        load_pkl = args.get("load_pkl", None)
        num_classes = args.get("num_classes", None)

        self.conv2d = getattr(slowfast, c2d_type)(slowfast_config=slowfast_config, slowfast_args=slowfast_args,
                                                  load_pkl=load_pkl, multi=True)

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, data):
        x = data['vid']

        if len(x.shape) == 5:
            framewise = self.conv2d(x.permute(0, 2, 1, 3, 4))
        else:
            # frame-wise features
            framewise = x
        return {
            "framewise_features": framewise,
        }


