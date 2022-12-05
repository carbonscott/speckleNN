import torch
import torch.nn as nn
from ..utils import ConvVolume, NNSize, TorchModelAttributeParser

from functools import reduce

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
        dim_img        = size_y * size_x
        isbias         = config.isbias
        dim_emb        = config.dim_emb

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

        # Define the embedding layer...
        self.embed = nn.Sequential(
            nn.Linear( in_features  = self.dim_features, 
                       out_features = 512, 
                       bias         = isbias),
            nn.PReLU(),
            nn.Linear( in_features  = 512, 
                       out_features = dim_emb, 
                       bias         = isbias),
        )


    def encode(self, x):
        x = self.conv(x)
        x = x.view(-1, self.dim_features)
        x = self.embed(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

        return x




class Hirotaka0122Plus(nn.Module):
    """ https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        dim_img        = size_y * size_x
        isbias         = config.isbias
        dim_emb        = config.dim_emb

        # Define Conv params...
        # I don't have a good idea of how to make it easier for users.
        conv_dict = {
                      "order"  : ("conv 1", "pool 1", 
                                  "conv 2", "pool 2",
                                  "conv 3", "pool 3"),
                      "conv 1" : { 
                                   "in_channels"  : 1,
                                   "out_channels" : 32,
                                   "kernel_size"  : 3,
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
                                   "kernel_size"  : 3,
                                   "stride"       : 1,
                                   "padding"      : 0,
                                   "bias"         : isbias,
                                 },
                      "pool 2" : { 
                                   "kernel_size" : 2,
                                   "stride"      : 2,
                                   "padding"     : 0,
                                 },
                      "conv 3" : { 
                                   "in_channels"  : 64,
                                   "out_channels" : 128,
                                   "kernel_size"  : 1,
                                   "stride"       : 1,
                                   "padding"      : 0,
                                   "bias"         : isbias,
                                 },
                      "pool 3" : { 
                                   "kernel_size" : 1,
                                   "stride"      : 1,
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

            nn.Conv2d(    **conv_dict["conv 3"] ),
            nn.PReLU(),
            nn.MaxPool2d( **conv_dict["pool 3"] ),
            nn.Dropout(0.1),
        )

        # Define the embedding layer...
        self.embed = nn.Sequential(
            nn.Linear( in_features  = self.dim_features, 
                       out_features = 512, 
                       bias         = isbias),
            nn.PReLU(),
            nn.Linear( in_features  = 512, 
                       out_features = dim_emb, 
                       bias         = isbias),
        )


    def encode(self, x):
        x = self.conv(x)
        x = x.view(-1, self.dim_features)
        x = self.embed(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

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

        # Define the embedding layer...
        self.embed = nn.Sequential(
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
        )


    def encode(self, x):
        x = self.conv(x)
        x = x.view(-1, self.dim_features)
        x = self.embed(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

        return x




class Shi2019(nn.Module):
    ''' DOI: 10.1107/S2052252519001854
    '''

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        isbias         = config.isbias

        # Define the feature extraction layer...
        in_channels = 1
        self.feature_extractor = nn.Sequential(
            # Motif network 1
            nn.Conv2d( in_channels  = in_channels,
                       out_channels = 5,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = isbias, ),
            nn.BatchNorm2d( num_features = 5 ),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d( kernel_size = 2, 
                          stride = 2 ),

            # Motif network 2
            nn.Conv2d( in_channels  = 5,
                       out_channels = 3,
                       kernel_size  = 3,
                       stride       = 1,
                       padding      = 0,
                       bias         = isbias, ),
            nn.BatchNorm2d( num_features = 3 ),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d( kernel_size = 2, 
                          stride = 2 ),

            # Motif network 3
            nn.Conv2d( in_channels  = 3,
                       out_channels = 2,
                       kernel_size  = 3,
                       stride       = 1,
                       padding      = 0,
                       bias         = isbias, ),
            nn.BatchNorm2d( num_features = 2 ),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d( kernel_size = 2, 
                          stride = 2 ),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in self.feature_extractor.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.feature_size = reduce(lambda x, y: x * y, NNSize(size_y, size_x, in_channels, conv_dict).shape())

        self.squash_to_logit = nn.Sequential(
            nn.Linear( in_features = self.feature_size,
                       out_features = 2,
                       bias = isbias ),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear( in_features = 2,
                       out_features = 1,
                       bias = isbias ),
            ## nn.Sigmoid(),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.feature_size)
        x = self.squash_to_logit(x)

        return x
