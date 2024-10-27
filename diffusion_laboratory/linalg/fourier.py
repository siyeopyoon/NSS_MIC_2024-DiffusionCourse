# random_tensor_laboratory/fourier.py
import torch

from .core import UnitaryLinearOperator, DiagonalLinearOperator, EigenDecomposedLinearOperator
    
class FourierTransform(UnitaryLinearOperator):
    def __init__(self, dim):
        """
        This class implements a N-Dimensional Fourier transform that can be used in a PyTorch model.

        it assumes the central pixel in the image is at the center of the input tensor in all dimensions

        it returns the Fourier transform with the zero-frequency component in the center of the image
        """
        super(FourierTransform, self).__init__()
        self.dim = dim

    def forward(self, x):
        x_ifftshift = torch.fft.ifftshift(x, dim=self.dim)
        x_fft = torch.fft.fftn(x_ifftshift, dim=self.dim, norm="ortho")
        x_fftshift = torch.fft.fftshift(x_fft, dim=self.dim)
        return x_fftshift
    
    def transpose(self, y):
        return torch.conj(self.conjugate_transpose( torch.conj(y)))

    def conjugate_transpose(self, y):
        y_ifftshift = torch.fft.ifftshift(y, dim=self.dim)
        y_ifft = torch.fft.ifftn(y_ifftshift, dim=self.dim, norm="ortho")
        y_fftshift = torch.fft.fftshift(y_ifft, dim=self.dim)
        return y_fftshift

class FourierLinearOperator(EigenDecomposedLinearOperator):
    def __init__(self, filter, dim):
        """
        This class implementts a ND Fourier filter that can be used in a PyTorch model.

        it assumes the central pixel in the image is at the center of the input tensor in all dimensions

        it returns the Fourier filter applied to the input. 
        """
        eigenvector_matrix = FourierTransform(dim=dim)
        eigenvalue_matrix = DiagonalLinearOperator(filter)
        super(FourierLinearOperator, self).__init__(eigenvector_matrix, eigenvalue_matrix)
        self.dim = dim
        self.filter = filter

    def mat_add(self, added_fourier_filter):
        assert isinstance(added_fourier_filter, (FourierLinearOperator)), "FourierLinearOperator addition only supported for FourierLinearOperator." 
        assert self.dim == added_fourier_filter.dim, "FourierLinearOperator addition only supported for FourierLinearOperator with same dim."
        return FourierLinearOperator(self.filter + added_fourier_filter.filter, dim=self.dim)
    
    def mat_sub(self, sub_fourier_filter):
        assert isinstance(sub_fourier_filter, (FourierLinearOperator)), "FourierLinearOperator subtraction only supported for FourierLinearOperator."
        assert self.dim == sub_fourier_filter.dim, "FourierLinearOperator subtraction only supported for FourierLinearOperator with same dim."
        return FourierLinearOperator(self.filter - sub_fourier_filter.filter, dim=self.dim)
    
    def mat_mul(self, mul_fourier_filter):
        assert isinstance(mul_fourier_filter, (FourierLinearOperator)), "FourierLinearOperator multiplication only supported for FourierLinearOperator."
        assert self.dim == mul_fourier_filter.dim, "FourierLinearOperator multiplication only supported for FourierLinearOperator with same dim."
        return FourierLinearOperator(self.filter * mul_fourier_filter.filter, dim=self.dim)
    

class FourierConvolution(FourierLinearOperator):
    def __init__(self, kernel, dim):
        """
        This class implements a 2D Fourier convolution that can be used in a PyTorch model.

        it assumes the central pixel in the image is at (0,0), including for the input kernel

        it returns the Fourier transform with the zero-frequency component in the center of the image
        """
        filter = FourierTransform(dim=dim).forward(kernel)
        super(FourierConvolution, self).__init__(filter, dim)
