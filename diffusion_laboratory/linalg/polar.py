
import torch

from .core import RealLinearOperator
from .interp import NearestNeighborInterpolator, BilinearInterpolator, LanczosInterpolator 


class PolarCoordinateResampler(RealLinearOperator):
    def __init__(self, theta_values, radius_values, row_center=None, col_center=None, interpolator=None):
        """
        This class implements a polar coordinate transformation that can be used in a PyTorch model.

        parameters:
            num_row: int
                The number of rows of the input image.
            num_col: int
                The number of columns of the input image.
            theta_values: torch.Tensor of shape [num_theta]
                The theta values, in radians, of the polar grid.
            radius_values: torch.Tensor of shape [num_radius]
                The radius values, in pixels, of the polar grid.
        """                        

        # Store the theta and radius values
        self.theta_values = theta_values
        self.radius_values = radius_values

        # Store the number of theta and radius values
        self.num_theta = len(theta_values)
        self.num_radius = len(radius_values)

        # Calculate the center of the image so that it works for even and odd dimensions
        if row_center is None:
            row_center = (self.num_row - 1) // 2
        if col_center is None:
            col_center = (self.num_col - 1) // 2

        # Create a meshgrid of theta and radius values
        theta_mesh, radius_mesh = torch.meshgrid(theta_values, radius_values)

        # Convert meshgrid to row, col coordinates
        #   where col is the horizontal axis, increasing from left to right
        #   and row is the vertical axis, increasing from top to bottom
        x_mesh = radius_mesh * torch.cos(theta_mesh)
        y_mesh = radius_mesh * torch.sin(theta_mesh)
        row_mesh = row_center - y_mesh + 1
        col_mesh = col_center + x_mesh + 1

        # Flatten and stack to get interp_points with shape [num_points, 2]
        interp_points = torch.stack((row_mesh.flatten(), col_mesh.flatten()), dim=1)
        
        if interpolator is None:
            interpolator = 'lanczos'
        
        assert interpolator in ['nearest', 'bilinear','lanczos'], "The interpolator must be one of 'nearest', 'bilinear', or 'lanczos'."

        if interpolator == 'nearest':
            self.interpolator = NearestNeighborInterpolator(self.num_row, self.num_col, interp_points)
        elif interpolator == 'bilinear':
            self.interpolator = BilinearInterpolator(self.num_row, self.num_col, interp_points)
        elif interpolator == 'lanczos':
            self.interpolator = LanczosInterpolator(self.num_row, self.num_col, interp_points, kernel_size=5)

        # Store the interp points
        self.interp_points = interp_points

        # Store shape for reshaping in forward method
        self.theta_mesh = theta_mesh
        self.radius_mesh = radius_mesh

    def forward(self, x):
        """
        This method implements the forward pass of the polar coordinate transformation.

        parameters:
            x: torch.Tensor 
                The input image to which the polar coordinate transformation should be applied.
        returns:
            result: torch.Tensor 
                The result of applying the polar coordinate transformation to the input image.
        """

        # Interpolate the values
        interpolated = self.interpolator(x)
        
        # Reshape to the original theta, r grid
        result = interpolated.view(*interpolated.shape[:2], *self.theta_mesh.shape)
        
        return result
    
    def transpose(self, y):
        """
        This method implements the transpose pass of the polar coordinate transformation.

        parameters:
            y: torch.Tensor 
                The input image to which the adjoint polar coordinate transformation should be applied.
        returns:
            result: torch.Tensor 
                The result of applying the adjoint polar coordinate transformation to the input image.
        """

        # Flatten the polar image
        flattened = y.view(*y.shape[:2], -1)
        
        # Use the adjoint method of the interpolator
        result = self.interpolator.transpose(flattened)
        
        return result