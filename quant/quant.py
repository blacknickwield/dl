from typing import Any, Union
import torch
from torch import Tensor
from torch import nn
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F
import math

class QuantHandler(torch.autograd.Function):
    def __init__(self, bits: int, *args, **kwargs) -> None:
        super(QuantHandler, self).__init__(*args, **kwargs)
        self.bits = bits
        self.q_min = -(2 ** (bits - 1))
        self.q_max = 2 ** (bits - 1) - 1
        self.r_max = None
        self.r_min = None

    @staticmethod
    def forward(ctx: Any, X, scale, zero, q_max, q_min, *args: Any, **kwargs: Any) -> Any:
        quant_X = zero + X / scale
        quant_X = quant_X.clamp(q_min, q_max).round_()
        dequant_X = scale * (quant_X - zero)

        return dequant_X
    
    @staticmethod
    def backward(ctx: Any, grad_output) -> Any:
        return grad_output, None, None, None, None


    def update(self, X: Tensor):
        self.r_max = X.max() if self.r_max is None else max(self.r_max, X.max())
        self.r_min = X.min() if self.r_min is None else min(self.r_min, X.min())

    def quantize(self, X: Tensor) -> Tensor:
        scale = float(self.r_max - self.r_min) / float(self.q_max - self.q_min)
        zero = int(self.q_max - self.r_max / scale)
        # TODO: clamp zero
        quant_X = QuantHandler.apply(X, scale, zero, self.q_max, self.q_min)
        return quant_X
    
    def dequantize(self, X: Tensor) -> Tensor:
        scale = float(self.r_max - self.r_min) / float(self.q_max - self.q_min)
        zero = int(self.q_max - self.r_max / scale)

        return scale * (X - zero)
    
    
class QuantLinear(nn.Linear):
    def __init__(self, bits: int, quantize_input: bool=False, quantize_weight: bool=True, *args, **kwargs) -> None:
        super(QuantLinear, self).__init__(*args, **kwargs)
        self.quantize_input = quantize_input
        self.quantize_weight = quantize_weight
        self.input_quant_handler = QuantHandler(bits=bits) if quantize_input  else None
        self.weight_quant_handler = QuantHandler(bits=bits) if quantize_weight else None
        
        self.do_collect = False
        self.do_quant = False

    def collect(self):
        self.do_collect = True
        self.do_quant = False

    def quant(self):
        self.do_collect = False
        self.do_quant = True    

    def forward(self, X) -> Tensor:
        if self.do_collect:
            if self.quantize_input:
                self.input_quant_handler.update(X=X)
            if self.quantize_weight:
                self.weight_quant_handler.update(X=X)
        
        if self.do_quant:
            quant_X = self.input_quant_handler.quantize(X=X) if self.quantize_input else X
            quant_weight = self.weight_quant_handler.quantize(X=self.weight) if self.quantize_weight else self.weight

            return F.linear(input=quant_X, weight=quant_weight, bias=self.bias)
                

        return F.linear(input=X, weight=self.weight, bias=self.bias)
    

class QuantConv2d(nn.Conv2d):
    def __init__(self, bits: int, quantize_input: bool=False, quantize_weight: bool=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bits = bits
        self.quantize_input = quantize_input
        self.quantize_weight = quantize_weight
        self.input_quant_handler = QuantHandler(bits=bits) if quantize_input else None
        self.weight_quant_handler = QuantHandler(bits=bits) if quantize_weight else None

        self.do_collect = False
        self.do_quant = False

    def collect(self):
        self.do_collect = True
        self.do_quant = False

    def quant(self):
        self.do_collect = False
        self.do_quant = True   
        
    def forward(self, X):
        if self.do_collect:
            if self.quantize_input:
                self.input_quant_handler.update(X=X)
            if self.quantize_weight:
                self.weight_quant_handler.update(X=X)

        if self.do_quant:
            quant_X = self.input_quant_handler.quantize(X=X) if self.quantize_input else X
            quant_weight = self.weight_quant_handler.quantize(X=self.weight) if self.quantize_weight else self.quant_weight
            
            return F.conv2d(input=quant_X, weight=quant_weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return F.conv2d(input=X, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

if __name__ == '__main__':
    x = 112.3