import torch

from .core import RealLinearOperator
    
class RealPart(RealLinearOperator):
    def __init__(self):
        """
        This class converts a complex tensor to a real tensor by taking the real part

        This is not truly a linear operator, but it can be used to convert a real-valued complex64 tensor to a float32 tensor

        """
        super(RealPart, self).__init__()

    def forward(self, x):
        return x.real
        
    def transpose(self, y):
        return y.real