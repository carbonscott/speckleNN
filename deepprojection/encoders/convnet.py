import torch
import torch.nn as nn
from deepprojection.utils import ConvVolume

import logging

logger = logging.getLogger(__name__)

class ConfigEncoder:

    def __init__(self, **kwargs):
        logger.info(f"___/ Configure Encoder \___")

        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16s} : {v}")


class Hirotaka0122(nn.Module):
    """ https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        isbias         = config.isbias

        dim_emb        = config.dim_emb
        dim_img = size_y * size_x

        # Define Conv params...
        # I don't have a good idea of how to make it easier for users.
        conv_dict = {
                      "order"  : ("conv 1", "pool 1", "conv 2", "pool 2"),
                      "conv 1" : { 
                                   "in_channels"  : 1,
                                   "out_channels" : 32,
                                   "kernel_size"  : 5,
                                   "stride"       : 1,
                                   "padding"      : 0,
                                   "bias"         : isbias,
                                 },
                      "pool 1" : { 
                                   "kernel_size" : 2,
                                   "stride"      : 2,
                                   "padding"     : 0,
                                 },
                      "conv 2" : { 
                                   "in_channels"  : 32,
                                   "out_channels" : 64,
                                   "kernel_size"  : 5,
                                   "stride"       : 1,
                                   "padding"      : 0,
                                   "bias"         : isbias,
                                 },
                      "pool 2" : { 
                                   "kernel_size" : 2,
                                   "stride"      : 2,
                                   "padding"     : 0,
                                 },
                    }

        in_channels = 1
        conv_volume = ConvVolume(size_y, size_x, in_channels, conv_dict)
        conv_channels, conv_size_y, conv_size_x = conv_volume.shape()
        self.dim_features = conv_size_y * conv_size_x * conv_channels

        # Define conv object...
        self.conv = nn.Sequential(
            # Conv layer for feature extraction...
            nn.Conv2d(    **conv_dict["conv 1"] ),
            nn.PReLU(),
            nn.MaxPool2d( **conv_dict["pool 1"] ),
            nn.Dropout(0.1),

            nn.Conv2d(    **conv_dict["conv 2"] ),
            nn.PReLU(),
            nn.MaxPool2d( **conv_dict["pool 2"] ),
            nn.Dropout(0.1),
        )

        # Define the scoring layer (classifer)...
        self.classifer = nn.Sequential(
            # Fully-connected layers for scoring...
            nn.Linear( in_features  = self.dim_features, 
                       ## out_features = dim_emb, 
                       out_features = 512, 
                       bias         = isbias),
            nn.PReLU(),
            ## nn.Linear( in_features  = dim_emb, 
            nn.Linear( in_features  = 512, 
                       out_features = dim_emb, 
                       bias         = isbias),
            ## nn.LogSoftmax(dim = -1)
        )


    def encode(self, x):
        x = self.conv(x)
        x = x.view(-1, self.dim_features)
        x = self.classifer(x)

        ## # L2 Normalize...
        ## dnorm = torch.norm(x)
        ## x = x / dnorm

        return x




class AdamBielski(nn.Module):
    """ https://github.com/adambielski/siamese-triplet/blob/master/networks.py """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        isbias         = config.isbias

        dim_emb        = config.dim_emb
        dim_img = size_y * size_x

        # Define Conv params...
        # I don't have a good idea of how to make it easier for users.
        conv_dict = {
                      "order"  : ("conv 1", "pool 1", "conv 2", "pool 2"),
                      "conv 1" : { 
                                   "in_channels"  : 1,
                                   "out_channels" : 32,
                                   "kernel_size"  : 5,
                                   "stride"       : 1,
                                   "padding"      : 0,
                                   "bias"         : isbias,
                                 },
                      "pool 1" : { 
                                   "kernel_size" : 2,
                                   "stride"      : 2,
                                   "padding"     : 0,
                                 },
                      "conv 2" : { 
                                   "in_channels"  : 32,
                                   "out_channels" : 64,
                                   "kernel_size"  : 5,
                                   "stride"       : 1,
                                   "padding"      : 0,
                                   "bias"         : isbias,
                                 },
                      "pool 2" : { 
                                   "kernel_size" : 2,
                                   "stride"      : 2,
                                   "padding"     : 0,
                                 },
                    }

        in_channels = 1
        conv_volume = ConvVolume(size_y, size_x, in_channels, conv_dict)
        conv_channels, conv_size_y, conv_size_x = conv_volume.shape()
        self.dim_features = conv_size_y * conv_size_x * conv_channels

        # Define conv object...
        self.conv = nn.Sequential(
            # Conv layer for feature extraction...
            nn.Conv2d(    **conv_dict["conv 1"] ),
            nn.PReLU(),
            nn.MaxPool2d( **conv_dict["pool 1"] ),
            nn.Dropout(0.1),

            nn.Conv2d(    **conv_dict["conv 2"] ),
            nn.PReLU(),
            nn.MaxPool2d( **conv_dict["pool 2"] ),
            nn.Dropout(0.1),
        )

        # Define the scoring layer (classifer)...
        self.classifer = nn.Sequential(
            # Fully-connected layers for scoring...
            nn.Linear( in_features  = self.dim_features, 
                       out_features = 256, 
                       bias         = isbias),
            nn.PReLU(),
            nn.Linear( in_features  = 256, 
                       out_features = 256, 
                       bias         = isbias),
            nn.PReLU(),
            nn.Linear( in_features  = 256, 
                       out_features = dim_emb, 
                       bias         = isbias),
            ## nn.LogSoftmax(dim = -1)
        )


    def encode(self, x):
        x = self.conv(x)
        x = x.view(-1, self.dim_features)
        x = self.classifer(x)

        return x
