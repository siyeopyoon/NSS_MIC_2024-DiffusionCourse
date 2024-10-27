
import torch

from .core import RealLinearOperator

class Pad(RealLinearOperator):
    def __init__(self, input_shape, pad_width, mode='constant', value=0):
        """
        This class implements a padding operator that can be used in a PyTorch model.

        it assumes the central pixel in the image is at (0,0)

        it returns the padded input
        """

        output_shape = tuple([input_shape[i] + pad_width[i] + pad_width[i + len(input_shape)] for i in range(len(input_shape))])
        super(Pad, self).__init__(input_shape, output_shape)
        self.pad_width = pad_width
        self.mode = mode
        self.value = value

    def forward(self, x):
        return torch.nn.functional.pad(x, self.pad_width, mode=self.mode, value=self.value)

    def transpose(self, y):
        return torch.nn.functional.pad(y, [-self.pad_width[i] for i in range(len(self.pad_width))], mode=self.mode, value=self.value)
