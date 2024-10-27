import torch

from .sparse import ColSparseLinearOperator

class NearestNeighborInterpolator(ColSparseLinearOperator):
    def __init__(self, num_row, num_col, interp_points):
        """
        This class implements a nearest neighbor interpolator that can be used in a PyTorch model.
        
        It inherits from the ColSparseLinearOperator class.
        
        The weights are precomputed using nearest neighbor interpolation.
        
        Parameters:
            num_row: int
                The number of rows of the input image.
            num_col: int
                The number of columns of the input image.
            interp_points: torch.Tensor of shape [num_points, 2]
                The coordinates of the interpolation points.
        """
        
        # Store the number of rows and columns
        self.num_row = num_row
        self.num_col = num_col
        
        # Extract x and y coordinates of interp_points
        ix = interp_points[:, 0]
        iy = interp_points[:, 1]
        
        # Calculate the nearest indices for interpolation
        self.ix0 = torch.round(ix).long()
        self.iy0 = torch.round(iy).long()
        
        # mask for out-of-boundary indices
        self.oob_mask = (self.ix0 >= self.num_row) | (self.iy0 >= self.num_col) | (self.ix0 < 0) | (self.iy0 < 0)
        self.ix0[self.oob_mask] = 0
        self.iy0[self.oob_mask] = 0
        
        # Calculate weights for each value
        self.wa = torch.ones_like(self.ix0).type(torch.float32)
        self.wa[self.oob_mask] = 0
        
        indices = self._2d_to_1d_indices(self.ix0, self.iy0).view(1, -1)
        weights = self.wa.flatten().view(1, -1)
        
        super(NearestNeighborInterpolator, self).__init__((num_row, num_col), (interp_points.shape[0],), indices, weights)
        
    def _2d_to_1d_indices(self, ix, iy):
        return ix * self.num_col + iy
    
    def _1d_to_2d_indices(self, index):
        ix = index // self.num_col
        iy = index % self.num_col
        return ix, iy
    
    def to(self, *args, **kwargs):
        """Override the to method to ensure that the a parameter is moved to the new device."""
        result = super().to(*args, **kwargs)
        return result

class BilinearInterpolator(ColSparseLinearOperator):
    def __init__(self, num_row, num_col, interp_points):

        """
        This class implements a bilinear interpolator that can be used in a PyTorch model.

        It inherits from the ColSparseLinearOperator class.

        The weights are precomputed using bilinear interpolation.

        parameters:
            num_row: int
                The number of rows of the input image.
            num_col: int
                The number of columns of the input image.
            interp_points: torch.Tensor of shape [num_points, 2]
                The coordinates of the interpolation points.
        """

        # Store the number of rows and columns
        self.num_row = num_row
        self.num_col = num_col
        
        # Extract x and y coordinates of interp_points
        ix = interp_points[:, 0]
        iy = interp_points[:, 1]

        # Calculate the four nearest indices for interpolation
        self.ix0 = torch.floor(ix).long()
        self.ix1 = self.ix0 + 1
        self.iy0 = torch.floor(iy).long()
        self.iy1 = self.iy0 + 1

        # mask for out-of-boundary indices
        self.oob_mask = (self.ix1 >= self.num_row-1) | (self.iy1 >= self.num_col-1) | (self.ix0 < 0) | (self.iy0 < 0)
        self.ix0[self.oob_mask] = 0
        self.ix1[self.oob_mask] = 0
        self.iy0[self.oob_mask] = 0
        self.iy1[self.oob_mask] = 0

        # Calculate weights for each value
        self.wa = (self.ix1.type(torch.float32) - ix) * (self.iy1.type(torch.float32) - iy)
        self.wb = (ix - self.ix0.type(torch.float32)) * (self.iy1.type(torch.float32) - iy)
        self.wc = (self.ix1.type(torch.float32) - ix) * (iy - self.iy0.type(torch.float32))
        self.wd = (ix - self.ix0.type(torch.float32)) * (iy - self.iy0.type(torch.float32))

        self.wa[self.oob_mask] = 0
        self.wb[self.oob_mask] = 0
        self.wc[self.oob_mask] = 0
        self.wd[self.oob_mask] = 0

        indices = torch.stack([self._2d_to_1d_indices(self.ix0, self.iy0),
                               self._2d_to_1d_indices(self.ix1, self.iy0),
                               self._2d_to_1d_indices(self.ix0, self.iy1),
                               self._2d_to_1d_indices(self.ix1, self.iy1)])
        
        weights = torch.stack([self.wa.flatten(), self.wb.flatten(), self.wc.flatten(), self.wd.flatten()])

        super(BilinearInterpolator, self).__init__((num_row, num_col), (interp_points.shape[0],), indices, weights)
        
    def _2d_to_1d_indices(self, ix, iy):
        return ix * self.num_col + iy
    
    def _1d_to_2d_indices(self, index):
        ix = index // self.num_col
        iy = index % self.num_col
        return ix, iy
    
    def to(self, *args, **kwargs):
        """Override the to method to ensure that the a parameter is moved to the new device."""
        result = super().to(*args, **kwargs)
        return result


class LanczosInterpolator(ColSparseLinearOperator):
    def __init__(self, num_row, num_col, interp_points, kernel_size=3):
        """
        This class implements the Lanczos interpolation method that can be used in a PyTorch model.
        
        Lanczos resampling is a high-quality resampling method which uses the sinc function 
        for sharper downsampled images with less aliasing than simpler methods.
        
        Parameters:
            num_row: int
                The number of rows of the input image.
            num_col: int
                The number of columns of the input image.
            interp_points: torch.Tensor of shape [num_points, 2]
                The coordinates of the interpolation points.
            kernel_size: int, default=3
                The size of the kernel or filter.
        """

        # Ensure the kernel_size is odd for symmetric filters
        assert kernel_size % 2 == 1, "kernel_size should be odd."

        # The 'a' parameter for Lanczos can be inferred from the kernel size
        self.a = kernel_size / 2.0
        self.num_row = num_row
        self.num_col = num_col

        # Base indices
        ix = torch.floor(interp_points[:, 0]).long()
        iy = torch.floor(interp_points[:, 1]).long()
        
        # Lists to store all indices and weights
        indices_list = []
        weights_list = []

        # Using floor of a for iteration range
        range_limit = int(torch.floor(torch.tensor(self.a)).item())

        relative_indices = [(i, j) for i in range(-range_limit, range_limit + 1) for j in range(-range_limit, range_limit + 1)]

        for i, j in relative_indices:
            ix_current = ix + i
            iy_current = iy + j
            
            # Create OOB mask for the current indices
            oob_mask = (ix_current >= self.num_row) | (iy_current >= self.num_col) | (ix_current < 0) | (iy_current < 0)

            # Set OOB indices to 0
            ix_current[oob_mask] = 0
            iy_current[oob_mask] = 0
            
            dx = torch.abs(interp_points[:, 0] - ix_current)
            dy = torch.abs(interp_points[:, 1] - iy_current)
            
            weight_x = self._lanczos(dx)
            weight_y = self._lanczos(dy)
            weight = weight_x * weight_y

            # Normalize weights so they sum to one
            sum_weight = weight.sum(dim=0, keepdim=True)
            weights = weight / sum_weight

            # Zero out the weights for OOB points
            weight[oob_mask] = 0
            
            index = self._2d_to_1d_indices(ix_current, iy_current)
            
            indices_list.append(index)
            weights_list.append(weight)

        indices = torch.stack(indices_list, dim=0)
        weights = torch.stack(weights_list, dim=0)

        # Normalize weights so they sum to one
        sum_weights = weights.sum(dim=0, keepdim=True)
        sum_weights[sum_weights == 0] = 1e-4
        weights = weights / sum_weights
        
        super(LanczosInterpolator, self).__init__((num_row, num_col), (interp_points.shape[0],), indices, weights)

    def _2d_to_1d_indices(self, ix, iy):
        return ix * self.num_col + iy
    
    def _1d_to_2d_indices(self, index):
        ix = index // self.num_col
        iy = index % self.num_col
        return ix, iy

    def _lanczos(self, x):
        """Compute the Lanczos weight."""
        pi_val = torch.tensor(3.14159265358979323846)  # Define the value of pi
        
        zero_mask = (x == 0).float()
        greater_mask = (x > self.a).float()
        default_mask = 1 - zero_mask - greater_mask
        
        sinc_x = torch.sin(pi_val * x) / (pi_val * x)
        sinc_x_over_a = torch.sin(pi_val * x / self.a) / (pi_val * x / self.a)

        sinc_x[torch.isnan(sinc_x)] = 1
        sinc_x_over_a[torch.isnan(sinc_x_over_a)] = 1
        
        return (sinc_x * sinc_x_over_a) * default_mask + zero_mask
    

