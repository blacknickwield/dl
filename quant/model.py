import torch
from torch import nn
from torch import Tensor
from quant import QuantConv2d, QuantLinear, QuantHandler
import torch.nn.functional as F


class QuantNet(nn.Module):
    def __init__(self, bits: int, in_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quant_conv1 = QuantConv2d(bits=8, quantize_input=True, quantize_weight=True, in_channels=in_channels, out_channels=40, kernel_size=3, stride=1)
        self.quant_conv2 = QuantConv2d(bits=8, quantize_input=False, quantize_weight=True, in_channels=40, out_channels=40, kernel_size=3, stride=1)
        self.quant_fc = QuantLinear(bits=8, quantize_input=False, quantize_weight=True, in_features=40 * 5 * 5, out_features=10)
        self.do_collect = False
        self.do_quant = False

    def collect(self):
        self.do_collect = True
        self.do_quant = False
        self.quant_conv1.collect()
        self.quant_conv2.collect()
        self.quant_fc.collect()

    def quant(self):
        self.do_collect = False
        self.do_quant = True
        self.quant_conv1.quant()
        self.quant_conv2.quant()
        self.quant_fc.quant()
    
    def forward(self, X) -> Tensor:
        X = F.max_pool2d(F.relu(self.quant_conv1(X)), kernel_size=2, stride=2)
        X = F.max_pool2d(F.relu(self.quant_conv2(X)), kernel_size=2, stride=2)
        X = X.view(-1, 40 * 5 * 5)
        X = self.quant_fc(X)

        return X

if __name__ == '__main__':
    pass