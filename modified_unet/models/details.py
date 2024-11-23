from models.base_unet import BaseUNet
import torch.nn as nn

class DetailedUNet(BaseUNet):
    def __init__(self, in_channels=1, out_channels=1):
        super(DetailedUNet, self).__init__(in_channels, out_channels)
        self.encoder.add_module("batch_norm", nn.BatchNorm2d(128))
