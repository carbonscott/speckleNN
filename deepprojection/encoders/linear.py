import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

class ConfigEncoder:

    def __init__(self, **kwargs):
        logger.info(f"___/ Configure Encoder \___")

        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16s} : {v}")


class SimpleEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        size_y, size_x = config.size_y, config.size_x
        dim_emb        = config.dim_emb
        isbias         = config.isbias

        dim_img = size_y * size_x

        self.conv = nn.Sequential(
            nn.Linear( in_features  = dim_img, 
                       out_features = dim_emb, 
                       bias         = isbias),
            nn.ReLU(),
            nn.Linear( in_features  = dim_emb, 
                       out_features = dim_emb, 
                       bias         = isbias),
            ## nn.Sigmoid()
        )


    def encode(self, x):
        x = self.conv(x)

        return x

