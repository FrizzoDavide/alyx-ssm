from dataclasses import dataclass

from src.hyperparameters.base_hyperparameters import BaseHyperparameters


@dataclass
class CNNHyperparameters(BaseHyperparameters):
    num_layers: int
    kernel_size: int
    conv_stride: int
    initial_channel_size: int
    channels_factor: int
    dropout: float
    max_pool_size: int
    activation: str = "ReLU"
    normalize_model_outputs: bool = False
    use_global_avg_pooling: bool = True #only for TCN
    num_downsample_layers: int = 0  #only for TCN
