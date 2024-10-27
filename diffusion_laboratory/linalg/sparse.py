import torch
import torch.nn as nn

from .core import LinearOperator

from matplotlib import pyplot as plt

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = '1'


pi = 3.1415927410125732


class ColSparseLinearOperator(LinearOperator):
    def __init__(self, input_shape, output_shape, indices, weights):
        """
        This class implements a column sparse linear operator that can be used in a PyTorch model.

        Column sparse linear operators have a small number of non-zero weights for each output element. 


        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            output_shape: tuple of integers
                The shape of the output tensor, disregarding batch and channel dimensions.
            indices: torch.Tensor of shape [num_weights, *output_shape]
                The 1D indices of the flattened input tensor that each weight corresponds to.
            weights: torch.Tensor of shape [num_weights, *output_shape]
                The weights of the linear operator.
        """

        super(ColSparseLinearOperator, self).__init__(input_shape, output_shape)

        # Check that indices and weights have the same shape.
        assert indices.shape == weights.shape, "Indices and weights must have the same shape."
        
        self.indices = indices
        self.weights = weights

    def forward(self, x, conjugate=False):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """

        batch_size, num_channel = x.shape[:2]
        
        assert x.shape[2:] == self.input_shape, "Input tensor shape doesn't match the specified input shape."

        result = torch.zeros(batch_size, num_channel, *self.output_shape, dtype=x.dtype, device=x.device)
        
        # Flatten the input tensor
        x_flattened = x.view(batch_size, num_channel, -1)
        
        for i in range(self.indices.shape[0]):  # Loop over num_weights
            values = x_flattened[:, :, self.indices[i]]  # Adding an additional dimension for broadcasting
            
            if conjugate is False:
                result += self.weights[i].view(1,1,-1) * values
            else:
                result += self.weights[i].conj().view(1,1,-1) * values

        result = result.view(batch_size, num_channel, *self.output_shape)

        return result
    
    def conjugate(self, x):
        return self.forward(x, conjugate=True)

    def transpose(self, x, conjugate=False):
        """
        This method implements the adjoint pass of the linear operator, i.e. the matrix-vector product with the adjoint.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The result of applying the adjoint of the linear operator to the input tensor.
        """

        batch_size, num_channel = x.shape[:2]
        
        assert x.shape[2:] == self.output_shape, "Input tensor shape doesn't match the specified output shape."
        
        result = torch.zeros(batch_size, num_channel, *self.input_shape, dtype=x.dtype, device=x.device)
        
        # Flatten the adjoint result tensor
        result_flattened = result.view(batch_size, num_channel, -1)
        x_flattened = x.view(batch_size, num_channel, -1)

        for i in range(self.indices.shape[0]):
            for b in range(batch_size):
                for c in range(num_channel):
                    if conjugate is False:
                        result_flattened[b, c].index_add_(0, self.indices[i].flatten(), (x_flattened[b, c] * self.weights[i].flatten()))
                    else:
                        result_flattened[b, c].index_add_(0, self.indices[i].flatten(), (x_flattened[b, c] * self.weights[i].conj().flatten()))
        return result

    def to(self, *args, **kwargs):
        self.indices = self.indices.to(*args, **kwargs)
        self.weights = self.weights.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    




class RowSparseLinearOperator(LinearOperator):
    """
    This class implements a row-column sparse linear operator that can be used in a PyTorch model.

    Row-column sparse linear operators have a small number of non-zero weights for each input element.

    parameters:
        input_shape: tuple of integers
            The shape of the input tensor, disregarding batch and channel dimensions.
        output_shape: tuple of integers
            The shape of the output tensor, disregarding batch and channel dimensions.
        indices: torch.Tensor of shape [num_weights, *input_shape]
            The 1D indices of the flattened output tensor that each weight corresponds to.
    """
    def __init__(self, input_shape, output_shape, indices, weights):
        super(RowSparseLinearOperator, self).__init__(input_shape, output_shape)
        assert indices.shape == weights.shape, "Indices and weights must have the same shape."
        self.indices = indices
        self.weights = weights

    def forward(self, x, conjugate=False):
        batch_size, num_channel = x.shape[:2]
        assert x.shape[2:] == self.input_shape, "Input tensor shape doesn't match the specified input shape."

        result = torch.zeros(batch_size, num_channel, *self.output_shape, dtype=x.dtype, device=x.device)
        
        results_flattened = result.view(batch_size, num_channel, -1)
        x_flattened = x.view(batch_size, num_channel, -1)
        
        for i in range(self.indices.shape[0]):
            for b in range(batch_size):
                for c in range(num_channel):
                    if conjugate is False:
                        results_flattened[b, c].index_add_(0, self.indices[i].flatten(), (x_flattened[b, c] * self.weights[i].flatten()))
                    else:
                        results_flattened[b, c].index_add_(0, self.indices[i].flatten(), (x_flattened[b, c] * self.weights[i].conj().flatten()))
        return result.view(batch_size, num_channel, *self.output_shape)
    
    def conjugate(self, x):
        return self.forward(x, conjugate=True)

    def transpose(self, y, conjugate=False):

        batch_size, num_channel = y.shape[:2]
        
        assert y.shape[2:] == self.output_shape, "Input tensor shape to adjoint doesn't match the specified output_shape of the linear operator."

        result = torch.zeros(batch_size, num_channel, *self.input_shape, dtype=y.dtype, device=y.device)
        result_flattened = result.view(batch_size, num_channel, -1)
        
        # Flatten the input tensor
        y_flattened = y.view(batch_size, num_channel, -1)
        
        for i in range(self.indices.shape[0]):  # Loop over num_weights
            values = y_flattened[:, :, self.indices[i].flatten()]  # Adding an additional dimension for broadcasting
            if conjugate is False:
                result_flattened += self.weights[i].view(1,1,-1) * values
            else:
                result_flattened += self.weights[i].conj().view(1,1,-1) * values

        result = result_flattened.view(batch_size, num_channel, *self.input_shape)

        return result
    
    def conjugate_transpose(self, y):
        return self.transpose(y, conjugate=True)

    def to(self, *args, **kwargs):
        self.indices = self.indices.to(*args, **kwargs)
        self.weights = self.weights.to(*args, **kwargs)
        return super().to(*args, **kwargs)





# Not quite working yet. Would be great to add something to convert these to torch.sparse.coo_tensor


# def unflatten_index(indices, shape):
#     """
#     Convert flattened indices into multi-dimensional indices.

#     parameters:
#         indices: torch.Tensor
#             A 1D tensor of flattened indices.
#         shape: tuple of int
#             The shape of the multi-dimensional tensor these indices belong to.

#     returns:
#         multi_dim_indices: torch.Tensor of shape [len(indices), len(shape)]
#             A tensor of multi-dimensional indices.
#     """

#     indices = indices.unsqueeze(0)  # Add an extra dimension for iteration over divmod
#     dim_size = torch.tensor(shape[::-1]).cumprod(0)  # Calculate cumulative product for mod and div operations

#     # Handle the edge case where the tensor is 1D
#     if len(shape) == 1:
#         return indices.T

#     # Initialize the multi-dimensional indices with zeros
#     multi_dim_indices = torch.zeros((indices.shape[1], len(shape)), dtype=torch.long, device=indices.device)

#     for i in range(len(shape) - 1, 0, -1):
#         multi_dim_indices[:, i], indices = divmod(indices, dim_size[i-1])

#     # The remaining indices belong to the first dimension
#     multi_dim_indices[:, 0] = indices.squeeze()

#     return multi_dim_indices

# def to_coo_tensor(self):
#     # Precompute the size of the COO tensor
#     total_size = (torch.prod(torch.tensor(self.input_shape)).item(), torch.prod(torch.tensor(self.output_shape)).item())
    
#     # Lists to store COO data
#     all_indices = []
#     all_values = []

#     # Loop over weights
#     for i in range(self.indices.shape[0]):
#         # Convert the flattened indices to sub-indices
#         flat_output_indices = torch.arange(torch.prod(torch.tensor(self.output_shape)))
#         input_coords = self.unflatten_index(self.indices[i], self.input_shape)
#         output_coords = self.unflatten_index(flat_output_indices, self.output_shape)

#         coo_indices = torch.stack([input_coords, output_coords], dim=0)
#         coo_indices = coo_indices.reshape(2, -1)
        
#         # Flatten the weights for the current weight index
#         coo_values = self.weights[i].flatten()

#         # Append to the lists
#         all_indices.append(coo_indices)
#         all_values.append(coo_values)

#     # Convert lists to tensors
#     coo_indices_tensor = torch.cat(all_indices, dim=1)
#     coo_values_tensor = torch.cat(all_values, dim=0)

#     # Return the COO tensor
#     return torch.sparse_coo_tensor(coo_indices_tensor, coo_values_tensor, size=total_size)
