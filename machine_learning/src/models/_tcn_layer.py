import torch
from torch import nn

#chomp layer to ensure the output length matches the input length
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]
    

class TCNLayer(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 conv_stride: int,
                 dropout: float,
                 activation: str,
                 dilation: int):
        super().__init__()

        activation_klass = getattr(nn, activation)

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=conv_stride,
                               padding=(kernel_size - 1) * dilation,
                               dilation=dilation)
        self.chomp1 = Chomp1d((kernel_size - 1) * dilation)
        self.act1 = activation_klass()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=conv_stride,
                               padding=(kernel_size - 1) * dilation,
                               dilation=dilation)
        self.chomp2 = Chomp1d((kernel_size - 1) * dilation)
        self.act2 = activation_klass()
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)

        #residual connection
        self.downsample = nn.Conv1d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.bn1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)

# class TCNLayer(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         stride,
#         dilation,
#         dropout,
#         activation,
#         pool: bool
#     ):
#         super().__init__()

#         activation_klass = getattr(nn, activation)
#         padding = (kernel_size - 1) * dilation

#         self.conv1 = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=padding,
#             dilation=dilation
#         )
#         self.chomp1 = Chomp1d(padding)
#         self.act1 = activation_klass()
#         self.dropout1 = nn.Dropout(dropout)

#         self.conv2 = nn.Conv1d(
#             out_channels,
#             out_channels,
#             kernel_size,
#             stride=1,
#             padding=padding,
#             dilation=dilation
#         )
#         self.chomp2 = Chomp1d(padding)
#         self.act2 = activation_klass()
#         self.dropout2 = nn.Dropout(dropout)

#         self.pool = nn.AvgPool1d(2) if pool else None

#         self.downsample = (
#             nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
#             if in_channels != out_channels or stride != 1
#             else None
#         )

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.chomp1(out)
#         out = self.act1(out)
#         out = self.dropout1(out)

#         out = self.conv2(out)
#         out = self.chomp2(out)
#         out = self.act2(out)
#         out = self.dropout2(out)

#         if self.pool is not None:
#             out = self.pool(out)

#         res = x if self.downsample is None else self.downsample(x)
#         if self.pool is not None:
#             res = self.pool(res)

#         return torch.relu(out + res)
