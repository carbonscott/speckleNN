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


class Hirotaka0122Large(nn.Module):
    """ https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        dim_img        = size_y * size_x
        bias           = config.isbias
        dim_emb        = config.dim_emb

        # Define the feature extraction layer...
        in_channels = 1
        self.conv = nn.Sequential(
            # CNN motif 1...
            nn.Conv2d( in_channels  = 1,
                       out_channels = 32,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            ## nn.BatchNorm2d( num_features = 32 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),
            nn.Dropout(0.1),

            # CNN motif 2...
            nn.Conv2d( in_channels  = 32,
                       out_channels = 64,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            ## nn.BatchNorm2d( num_features = 64 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),
            nn.Dropout(0.1),

            # CNN motif 3...
            nn.Conv2d( in_channels  = 64,
                       out_channels = 128,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            ## nn.BatchNorm2d( num_features = 128 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),
            nn.Dropout(0.1),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in self.conv.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.feature_size = reduce(lambda x, y: x * y, NNSize(size_y, size_x, in_channels, conv_dict).shape())

        # Define the embedding layer...
        self.embed = nn.Sequential(
            nn.Linear( in_features  = self.feature_size, 
                       out_features = 512, 
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = 512, 
                       out_features = dim_emb, 
                       bias         = bias),
        )


    def encode(self, x):
        x = self.conv(x)
        x = x.view(-1, self.feature_size)
        x = self.embed(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

        return x




class Hirotaka0122(nn.Module):
    """ https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        dim_img        = size_y * size_x
        bias           = config.isbias
        dim_emb        = config.dim_emb

        # Define the feature extraction layer...
        in_channels = 1
        self.conv = nn.Sequential(
            # CNN motif 1...
            nn.Conv2d( in_channels  = 1,
                       out_channels = 32,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            ## nn.BatchNorm2d( num_features = 32 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),
            nn.Dropout(0.1),

            # CNN motif 2...
            nn.Conv2d( in_channels  = 32,
                       out_channels = 64,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            ## nn.BatchNorm2d( num_features = 64 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),
            nn.Dropout(0.1),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in self.conv.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.feature_size = reduce(lambda x, y: x * y, NNSize(size_y, size_x, in_channels, conv_dict).shape())

        # Define the embedding layer...
        self.embed = nn.Sequential(
            nn.Linear( in_features  = self.feature_size, 
                       out_features = 512, 
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = 512, 
                       out_features = dim_emb, 
                       bias         = bias),
        )


    def encode(self, x):
        x = self.conv(x)
        x = x.view(-1, self.feature_size)
        x = self.embed(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

        return x




class FewShotModel(nn.Module):
    """ ... """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        dim_img        = size_y * size_x
        bias           = config.isbias
        dim_emb        = config.dim_emb

        # Define the feature extraction layer...
        in_channels = 1
        self.feature_extractor = nn.Sequential(
            # CNN motif 1...
            nn.Conv2d( in_channels  = 1,
                       out_channels = 32,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.BatchNorm2d( num_features = 32 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 2...
            nn.Conv2d( in_channels  = 32,
                       out_channels = 64,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.BatchNorm2d( num_features = 64 ),
            nn.PReLU(),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in self.feature_extractor.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.feature_size = reduce(lambda x, y: x * y, NNSize(size_y, size_x, in_channels, conv_dict).shape())

        # Define the embedding layer...
        self.embed = nn.Sequential(
            nn.Linear( in_features  = self.feature_size, 
                       out_features = 512, 
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = 512, 
                       out_features = dim_emb, 
                       bias         = bias),
        )


    def encode(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.feature_size)
        x = self.embed(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

        return x




class FewShotModel2(nn.Module):
    """ ... """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        dim_img        = size_y * size_x
        bias           = config.isbias
        dim_emb        = config.dim_emb

        # Define the feature extraction layer...
        in_channels = 1
        self.feature_extractor = nn.Sequential(
            # CNN motif 1...
            nn.Conv2d( in_channels  = 1,
                       out_channels = 32,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 32 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 2...
            nn.Conv2d( in_channels  = 32,
                       out_channels = 64,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 64 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 3...
            nn.Conv2d( in_channels  = 64,
                       out_channels = 128,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 128 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in self.feature_extractor.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.feature_size = reduce(lambda x, y: x * y, NNSize(size_y, size_x, in_channels, conv_dict).shape())

        # Define the embedding layer...
        self.embed = nn.Sequential(
            nn.Linear( in_features  = self.feature_size, 
                       out_features = 512, 
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = 512, 
                       out_features = dim_emb, 
                       bias         = bias),
        )


    def encode(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.feature_size)
        x = self.embed(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

        return x




class FewShotModel3(nn.Module):
    """ ... """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        dim_img        = size_y * size_x
        bias           = config.isbias
        dim_emb        = config.dim_emb

        # Define the feature extraction layer...
        in_channels = 1
        self.feature_extractor = nn.Sequential(
            # CNN motif 1...
            nn.Conv2d( in_channels  = 1,
                       out_channels = 32,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 32 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 2...
            nn.Conv2d( in_channels  = 32,
                       out_channels = 64,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 64 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 3...
            nn.Conv2d( in_channels  = 64,
                       out_channels = 128,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 128 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in self.feature_extractor.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.feature_size = reduce(lambda x, y: x * y, NNSize(size_y, size_x, in_channels, conv_dict).shape())

        # Define the embedding layer...
        self.embed = nn.Sequential(
            nn.Linear( in_features  = self.feature_size,
                       out_features = self.feature_size // 3,
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = self.feature_size // 3,
                       out_features = self.feature_size // 9,
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = self.feature_size // 9,
                       out_features = dim_emb,
                       bias         = bias),
        )


    def encode(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.feature_size)
        x = self.embed(x)

        # L2 Normalize...
        dnorm = torch.norm(x, dim = -1, keepdim = True)
        x = x / dnorm

        return x




class FewShotModel4(nn.Module):
    """ ... """

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        dim_img        = size_y * size_x
        bias           = config.isbias
        dim_emb        = config.dim_emb

        # Define the feature extraction layer...
        in_channels = 1
        self.feature_extractor = nn.Sequential(
            # CNN motif 1...
            nn.Conv2d( in_channels  = 1,
                       out_channels = 32,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 32 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 2...
            nn.Conv2d( in_channels  = 32,
                       out_channels = 32,
                       kernel_size  = 1,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.Conv2d( in_channels  = 32,
                       out_channels = 64,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 64 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),

            # CNN motif 3...
            nn.Conv2d( in_channels  = 64,
                       out_channels = 64,
                       kernel_size  = 1,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.Conv2d( in_channels  = 64,
                       out_channels = 128,
                       kernel_size  = 5,
                       stride       = 1,
                       padding      = 0,
                       bias         = bias, ),
            nn.PReLU(),
            nn.BatchNorm2d( num_features = 128 ),
            nn.MaxPool2d( kernel_size = 2,
                          stride      = 2,
                          padding     = 0, ),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in self.feature_extractor.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.feature_size = reduce(lambda x, y: x * y, NNSize(size_y, size_x, in_channels, conv_dict).shape())

        # Define the embedding layer...
        self.embed = nn.Sequential(
            nn.Linear( in_features  = self.feature_size,
                       out_features = self.feature_size // 3,
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = self.feature_size // 3,
                       out_features = self.feature_size // 9,
                       bias         = bias),
            nn.PReLU(),
            nn.Linear( in_features  = self.feature_size // 9,
                       out_features = dim_emb,
                       bias         = bias),
        )


    def encode(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.feature_size)
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
