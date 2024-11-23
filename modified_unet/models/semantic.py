from models.base_unet import BaseUNet
import torch.nn as nn

class SemanticUNet(BaseUNet):
    def __init__(self, in_channels=1, out_channels=1):
        super(SemanticUNet, self).__init__(in_channels, out_channels)
        self.decoder.add_module("dropout", nn.Dropout2d(0.2))
