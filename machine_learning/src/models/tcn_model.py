import numpy as np
import torch
from torch import nn

from src.models._tcn_layer import TCNLayer
from src.hyperparameters.cnn_hyperparameters import CNNHyperparameters 


class TCNModel(nn.Module):
    def __init__(self, hyperparameters: CNNHyperparameters, num_features: int, window_size: int, num_out_classes: int):
        super().__init__()
        self.num_features = num_features
        self.num_out_classes = num_out_classes
        self.hparams = hyperparameters

        layers = []
        in_channels = num_features
        for layer_idx in range(self.hparams.num_layers):
            out_channels = int(self.hparams.initial_channel_size *
                               self.hparams.channels_factor**layer_idx)
            layers.append(TCNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.hparams.kernel_size,
                conv_stride=1,
                dropout=self.hparams.dropout,
                activation=self.hparams.activation,
                dilation=2**layer_idx
            ))
            in_channels = out_channels  #next layer's in_channels

        self.tcn_layers = nn.Sequential(*layers)

        #fc layer
        if self.hparams.use_global_avg_pooling:
            self.fc = nn.Linear(in_channels, num_out_classes)
        else:
            self.fc = nn.Linear(in_channels * window_size, num_out_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn_layers(x)

        if self.hparams.use_global_avg_pooling:
            x = torch.mean(x, dim=2)
        else:
            x = torch.flatten(x, start_dim=1)

        x = self.fc(x)
        return x

# class TCNModel(nn.Module):
#     def __init__(self, hyperparameters, num_features: int, window_size: int, num_out_classes: int):
#         super().__init__()

#         self.num_features = num_features
#         self.num_out_classes = num_out_classes
#         self.hparams = hyperparameters

#         layers = []
#         in_channels = num_features

#         for layer_idx in range(self.hparams.num_layers):
#             out_channels = int(
#                 self.hparams.initial_channel_size *
#                 self.hparams.channels_factor ** layer_idx
#             )

#             # 🔑 Downsample only in early layers
#             if layer_idx < self.hparams.num_downsample_layers:
#                 stride = 1
#                 pool = True
#             else:
#                 stride = 1
#                 pool = False

#             layers.append(
#                 TCNLayer(
#                     in_channels=in_channels,
#                     out_channels=out_channels,
#                     kernel_size=self.hparams.kernel_size,
#                     stride=stride,
#                     dilation=2 ** layer_idx,
#                     dropout=self.hparams.dropout,
#                     activation=self.hparams.activation,
#                     pool=pool
#                 )
#             )

#             in_channels = out_channels

#         self.tcn = nn.Sequential(*layers)
#         self.fc = nn.Linear(in_channels, num_out_classes)

#     def forward(self, x):
#         # (B, T, C) → (B, C, T)
#         x = x.transpose(1, 2)

#         x = self.tcn(x)

#         # global average pooling over time
#         x = x.mean(dim=2)

#         return self.fc(x)