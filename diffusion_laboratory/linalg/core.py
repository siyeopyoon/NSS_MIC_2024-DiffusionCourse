# random_tensor_laboratory/linalg/core.py
import torch

class LinearOperator(torch.nn.Module):
    def __init__(self):
        """
        This is an abstract class for linear operators.

        It inherits from torch.nn.Module.
        
        It requires the methods forward and adjoint to be implemented.

        parameters:
            None
        """

        super(LinearOperator, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor 
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """

        return NotImplementedError
    
    def forward_LinearOperator(self):
        return self
    
    def transpose(self,y: torch.Tensor) -> torch.Tensor:
        """
        This method returns the transpose of the linear operator.

        parameters:
            y: torch.Tensor
                The input tensor to the transpose of the linear operator.

        returns:
            z: torch.Tensor
                The transpose of the linear operator applied to y
        """
        
        # by default, use a forward pass applied to zeros and then transpose
        # this is not the most efficient way to do this, but it is the most general

        # check if self.input_shape is defined
        if hasattr(self, 'input_shape'):
            input_shape = self.input_shape
        else:
            if hasattr(self, 'input_shape_given_output_shape'):
                self.input_shape = self.input_shape_given_output_shape(y.shape)
            else:
                raise NotImplementedError("Subclass of rtl.linear_operators.LinearOperator must either define the input_shape, define the input_shape_given_output_shape(output_shape) method, or attribute or implement the transpose method.")

        _input = torch.zeros(input_shape, dtype=y.dtype, device=y.device)
        _input.requires_grad = True
        _output = self.forward(_input)
        # now use autograd to compute the transpose applied to y
        # we also need the transpose operation itself to be differentiable
        _output.backward(y, create_graph=True)
        # backpropagtion through the forward pass is the transpose pass
        z = _input.grad
        return z
    
    def transpose_LinearOperator(self):
        return TransposeLinearOperator(self)
    
    def conjugate(self, x: torch.Tensor):
        """
        This method returns the conjugate of the linear operator.

        parameters:
            x: torch.Tensor 
                The input tensor to the conjugate of the linear operator.

        returns:
            conjugate: LinearOperator object
                The conjugate of the linear operator.

        """
        return torch.conj(self.forward(torch.conj(x)))

    def conjugate_LinearOperator(self):
        return ConjugateLinearOperator(self)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the adjoint pass of the linear operator, i.e. the conjugate-transposed matrix-vector product.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to the adjoint of the linear operator.
        returns:
            z: torch.Tensor 
                The result of applying the adjoint of the linear operator to the input tensor.
        """

        z = torch.conj(self.transpose(torch.conj(y)))
        return z
    
    def conjugate_transpose_LinearOperator(self):
        return ConjugateTransposeLinearOperator(self)    
    
    def sqrt_LinearOperator(self):
        raise NotImplementedError
    
    def inv_LinearOperator(self):
        return InverseLinearOperator(self)
    
    def logdet(self):
        raise NotImplementedError
    
    def det(self):
        return torch.exp(self.logdet())
    
    def _pseudoinverse_weighted_average(self, y: torch.Tensor):
        """
        This method implements the pseudoinverse of the linear operator using a weighted average.

        parameters:
            y: torch.Tensor 
                The input tensor to the pseudoinverse_weighted_average of the linear operator.
        returns:
            x: torch.Tensor 
                The result of applying the pseudoinverse_weighted_average of the linear operator to the input tensor.
        """
        
        numerator = self.conjugate_transpose(y)
        
        denominator = self.conjugate_transpose(torch.ones_like(y))

        x = numerator / (denominator + 1e-10)  # Avoid division by zero
        
        return x

    def _pseudoinverse_conjugate_gradient(self, b, max_iter=1000, tol=1e-6, beta=1e-3, verbose=False):
        """
        This method implements the pseudoinverse of the linear operator using the conjugate gradient method.

        It solves the linear system (A^T A + beta * I) x = A^T b for x, where A is the linear operator.

        parameters:
            b: torch.Tensor of shape
                The input tensor to which the pseudo inverse of the linear operator should be applied.
            max_iter: int
                The maximum number of iterations to run the conjugate gradient method.
            tol: float
                The tolerance for the conjugate gradient method.
            beta: float
                The regularization strength for the conjugate gradient method.
        returns:
            x_est: torch.Tensor
                The result of applying the pseudoinverse_conjugate_gradient of the linear operator to the input tensor.
        """
        ATb = self.conjugate_transpose(b)
        x_est = self._pseudo_inverse_weighted_average(b)
        
        r = ATb - self.conjugate_transpose(self.forward(x_est)) - beta * x_est
        p = r.clone()
        rsold = torch.dot(r.flatten(), r.flatten())
        
        for i in range(max_iter):
            if verbose:
                print("Inverting ", self.__class__.__name__, " with conjugate_gradient. Iteration: {}, Residual: {}".format(i, torch.sqrt(torch.abs(rsold))))
            ATAp = self.conjugate_transpose(self.forward(p)) + beta * p
            alpha = rsold / torch.dot(p.flatten(), ATAp.flatten())
            x_est += alpha * p
            r -= alpha * ATAp
            rsnew = torch.dot(r.flatten(), r.flatten())
            if torch.sqrt(torch.abs(rsnew)) < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            
        return x_est
    
    def pseudoinverse(self, y, method=None, **kwargs):
        """
        This method implements the pseudo inverse of the linear operator.

        parameters:
            y: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The input tensor to which the pseudo inverse of the linear operator should be applied.
            method: str
                The method to use for computing the pseudo inverse. If None, the method is chosen automatically.
            kwargs: dict
                Keyword arguments to be passed to the method.
        """

        if method is None:
            method = 'conjugate_gradient'

        assert method in ['weighted_average', 'conjugate_gradient'], "The method should be either 'weighted_average' or 'conjugate_gradient'."

        if method == 'weighted_average':
            result =  self._pseudoinverse_weighted_average(y, **kwargs)
        elif method == 'conjugate_gradient':
            result =  self._pseudoinverse_conjugate_gradient(y, **kwargs)

        return result
    
    def mat_add(self, M):
        raise NotImplementedError
    def mat_sub(self, M):
        raise NotImplementedError
    def mat_mul(self, M):
        return self.forward(M)
    def __mul__(self, x):
        return NotImplementedError
    def __add__(self, M):
        return self.mat_add(M)
    def __sub__(self, M):
        return self.mat_sub(M)
    def __matmul__(self, M):
        return self.mat_mul(M)

    
class RealLinearOperator(LinearOperator):
    def __init__(self):
        """
        This is an abstract class for real linear operators.

        It inherits from LinearOperator.

        parameters:
            None
        """

        super(RealLinearOperator, self).__init__()

    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        """
        This method returns the conjugate of the linear operator.

        parameters:
            x: torch.Tensor
                The input tensor to the conjugate of the linear operator.

        returns:
            conjugate: LinearOperator object
                The conjugate of the linear operator.

        """
        # for real linear operators, the conjugate is the same as the forward
        self.forward(x)        
    
    def conjugate_LinearOperator(self):
        return self.forward_LinearOperator()
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """ 
        This method returns the conjugate transpose of the linear operator.

        parameters:
            y: torch.Tensor
                The input tensor to the conjugate_transpose of the linear operator.

        returns:
            z: torch.Tensor
                The conjugate_transpose of the linear operator.

        """
        # for real linear operators, the conjugate transpose is the same as the transpose
        return self.transpose(y)
    
    def conjugate_transpose_LinearOperator(self):
        return self.transpose_LinearOperator()
    

class SquareLinearOperator(LinearOperator):
    def __init__(self):
        """
        This is an abstract class for square linear operators.

        It inherits from LinearOperator.

        For square linear operators, the input and output shapes are the same.

        parameters:
            None
        """

        super(SquareLinearOperator, self).__init__()

    def compute_input_shape_given_output_shape(self, output_shape):
        """
        This method computes the input shape given the output shape.

        parameters:
            output_shape: tuple of integers
                The shape of the output tensor, disregarding batch and channel dimensions.
        returns:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
        """
        # for square linear operators, the input and output shapes are the same
        return output_shape
    

class InvertibleLinearOperator(SquareLinearOperator):
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear operator.

        parameters:
            y: torch.Tensor 
                The input tensor to which the inverse linear operator should be applied.
        returns:
            x: torch.Tensor
                The result of applying the inverse linear operator to the input tensor.
        """
        if not hasattr(self, 'inverse_LinearOperator'):
            return self.inverse_LinearOperator().forward(y)
        else:
            raise NotImplementedError("For InvrtibleLinearOperator, either the inverse method or the inverse_LinearOperator method must be implemented.")
    
    def inverse_LinearOperator(self):
        """
        This method returns the inverse of the linear operator.

        returns:
            inverse: LinearOperator object
                The inverse of the linear operator.
        """
        if not hasattr(self, 'inverse'):
            return InverseLinearOperator(self)
        else:
            raise NotImplementedError("For InvrtibleLinearOperator, either the inverse method or the inverse_LinearOperator method must be implemented.")
        
class UnitaryLinearOperator(InvertibleLinearOperator):
    def __init__(self):
        """
        This is an abstract class for unitary linear operators.

        It inherits from InvertibleLinearOperator.

        parameters:
            None
        """

        super(UnitaryLinearOperator, self).__init__()

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the inverse of the linear operator.

        parameters:
            y: torch.Tensor 
                The input tensor to which the inverse linear operator should be applied.
        returns:
            x: torch.Tensor
                The result of applying the inverse linear operator to the input tensor.
        """
        # for unitary linear operators, the inverse is the conjugate transpose
        return self.conjugate_transpose(y)

    def inverse_LinearOperator(self):
        return self.conjugate_transpose_LinearOperator()


class HermitianLinearOperator(SquareLinearOperator):
    def __init__(self):
        """
        This is an abstract class for Hermitian, or self-adjoint linear operators.

        It inherits from SquareLinearOperator.

        parameters:
            None
        """

        super(HermitianLinearOperator, self).__init__()

    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.forward(y)
    
    def conjugate_transpose_LinearOperator(self):
        return self
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.transpose(x)
    
    def conjugate_LinearOperator(self):
        return self.transpose_LinearOperator()
    

class SymmetricLinearOperator(SquareLinearOperator):
    def __init__(self):
        """
        This is an abstract class for Symmetric linear operators.

        It inherits from SquareLinearOperator.

        parameters:
            None
        """

        super(SymmetricLinearOperator, self).__init__()

    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the transpose of the linear operator.

        parameters:
            y: torch.Tensor
                The input tensor to the transpose of the linear operator.
        returns:
            z: torch.Tensor
                The result of applying the transpose of the linear operator to the input tensor.
        """
        return self.forward(y)
    
    def cojugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        """
        This method implements the adjoint pass of the linear operator, i.e. the matrix-vector product with the adjoint.

        parameters:
            y: torch.Tensor 
                The input tensor to the adjoint of the linear operator.
        returns:
            z: torch.Tensor 
                The result of applying the adjoint of the linear operator to the input tensor.
        """
        return self.conjugate(y)
    
    def transpose_LinearOperator(self):
        return self
    
class ScalarLinearOperator(SymmetricLinearOperator, InvertibleLinearOperator):
    def __init__(self, scalar):
        """
        This class implements a scalar linear operator.

        It inherits from SymmetricLinearOperator.

        parameters:
            scalar: float
                The scalar to multiply the input tensor with.
        """

        super(ScalarLinearOperator, self).__init__()

        # if scalar is a float, convert it to a tensor
        if isinstance(scalar, (int, float)):
            scalar = torch.tensor(scalar)

        self.scalar = scalar

    def forward(self, x):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor 
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor 
                The result of applying the linear operator to the input tensor.
        """

        return self.scalar * x
    
    def conjugate(self, x):
        """
        This method implements the adjoint pass of the linear operator, i.e. the conjugate-transposed matrix-vector product.

        parameters:
            x: torch.Tensor 
                The input tensor to the adjoint of the linear operator.
        returns:
            adj_result: torch.Tensor
                The result of applying the adjoint of the linear operator to the input tensor.
        """
        return torch.conj(self.scalar) * x
    
    def inverse(self, y):
        if self.scalar == 0:
            raise ValueError("The scalar is zero, so the inverse does not exist.")
        return y / self.scalar
    
    def inverse_LinearOperator(self):
        # if self.scalar == 0:
            # raise ValueError("The scalar is zero, so the inverse does not exist.")
        return ScalarLinearOperator(1/self.scalar)
    
    def sqrt_LinearOperator(self):
        return ScalarLinearOperator(torch.sqrt(self.scalar))

    def mat_add(self, added_scalar_matrix):
        assert isinstance(added_scalar_matrix, (ScalarLinearOperator)), "ScalarLinearOperator addition only supported for ScalarLinearOperator." 
        return ScalarLinearOperator(self.scalar + added_scalar_matrix.scalar)
    
    def mat_sub(self, sub_scalar_matrix):
        assert isinstance(sub_scalar_matrix, (ScalarLinearOperator)), "ScalarLinearOperator subtraction only supported for ScalarLinearOperator." 
        return ScalarLinearOperator(self.scalar - sub_scalar_matrix.scalar)

    def mat_mul(self, mul_scalar_matrix):
        if isinstance(mul_scalar_matrix, torch.Tensor):
            return self.forward(mul_scalar_matrix)
        elif isinstance(mul_scalar_matrix, ScalarLinearOperator):
            return ScalarLinearOperator(self.scalar * mul_scalar_matrix.scalar)
        else:
            raise ValueError("Unsupported type for matrix multiplication.")
    


class DiagonalLinearOperator(SquareLinearOperator):
    def __init__(self, diagonal_vector):
        """
        This class implements a diagonal linear operator.

        It inherits from SquareLinearOperator.

        parameters:
            input_shape: tuple of integers
                The shape of the input tensor, disregarding batch and channel dimensions.
            diagonal_vector: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The diagonal of the linear operator.
        """

        super(DiagonalLinearOperator, self).__init__()

        self.diagonal_vector = diagonal_vector
    
    def forward(self, x):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor of shape [batch_size, num_channel, *input_shape]
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor of shape [batch_size, num_channel, *output_shape]
                The result of applying the linear operator to the input tensor.
        """
        return self.diagonal_vector * x
    
    def conjugate(self, x):
        return torch.conj(self.diagonal_vector) * x
    
    def inverse_LinearOperator(self):
        if torch.any(self.diagonal_vector == 0):
            raise ValueError("The diagonal vector contains zeros, so the inverse does not exist.")
        return DiagonalLinearOperator(self.input_shape, 1/self.diagonal_vector)
    
    def sqrt_LinearOperator(self):
        return DiagonalLinearOperator(self.input_shape, torch.sqrt(self.diagonal_vector))

    def mat_add(self, added_diagonal_matrix):
        assert isinstance(added_diagonal_matrix, (DiagonalLinearOperator)), "DiagonalLinearOperator addition only supported for DiagonalLinearOperator." 
        assert self.input_shape == added_diagonal_matrix.input_shape, "DiagonalLinearOperator addition only supported for DiagonalLinearOperator with same input shape."
        return DiagonalLinearOperator(self.input_shape, self.diagonal_vector + added_diagonal_matrix.diagonal_vector)

    def mat_sub(self, sub_diagonal_matrix):
        assert isinstance(sub_diagonal_matrix, (DiagonalLinearOperator)), "DiagonalLinearOperator subtraction only supported for DiagonalLinearOperator." 
        assert self.input_shape == sub_diagonal_matrix.input_shape, "DiagonalLinearOperator subtraction only supported for DiagonalLinearOperator with same input shape."
        return DiagonalLinearOperator(self.input_shape, self.diagonal_vector - sub_diagonal_matrix.diagonal_vector)
    
    def mat_mul(self, mul_diagonal_matrix):
        assert isinstance(mul_diagonal_matrix, (DiagonalLinearOperator)), "DiagonalLinearOperator multiplication only supported for DiagonalLinearOperator." 
        assert self.input_shape == mul_diagonal_matrix.input_shape, "DiagonalLinearOperator multiplication only supported for DiagonalLinearOperator with same input shape."
        return DiagonalLinearOperator(self.input_shape, self.diagonal_vector * mul_diagonal_matrix.diagonal_vector)


class IdentityLinearOperator(RealLinearOperator, UnitaryLinearOperator, HermitianLinearOperator, SymmetricLinearOperator):
    def __init__(self):
        """
        This class implements the identity linear operator.

        It inherits from SquareLinearOperator.

        parameters:
            None
        """

        SquareLinearOperator.__init__(self)

    def forward(self, x):
        """
        This method implements the forward pass of the linear operator, i.e. the matrix-vector product.

        parameters:
            x: torch.Tensor 
                The input tensor to the linear operator.
        returns:
            result: torch.Tensor 
                The result of applying the linear operator to the input tensor.
        """
        return x
    



class ConjugateLinearOperator(LinearOperator):
    def __init__(self, base_matrix_operator: LinearOperator):
        """
        This is an abstract class for linear operators that are the conjugate of another linear operator.

        It inherits from LinearOperator.

        parameters:
            base_matrix_operator: LinearOperator object
                The linear operator to which the conjugate should be applied.
        """
            
        assert isinstance(base_matrix_operator, LinearOperator), "The linear operator should be a LinearOperator object."
        super(ConjugateLinearOperator, self).__init__(base_matrix_operator.output_shape, base_matrix_operator.input_shape)

        self.base_matrix_operator = base_matrix_operator  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(y)
    
class TransposeLinearOperator(LinearOperator):
    def __init__(self, base_matrix_operator: LinearOperator):
        """
        This is an abstract class for linear operators that are the transpose of another linear operator.

        It inherits from LinearOperator.

        parameters:
            base_matrix_operator: LinearOperator object
                The linear operator to which the conjugate should be applied.
        """
            
        assert isinstance(base_matrix_operator, LinearOperator), "The linear operator should be a LinearOperator object."

        super(TransposeLinearOperator, self).__init__()

        self.base_matrix_operator = base_matrix_operator  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(y)

class ConjugateTransposeLinearOperator(LinearOperator):
    def __init__(self, base_matrix_operator: LinearOperator):
        """
        This is an abstract class for linear operators that are the conjugate transpose of another linear operator.

        It inherits from LinearOperator.

        parameters:
            base_matrix_operator: LinearOperator object
                The linear operator to which the conjugate should be applied.
        """
            
        assert isinstance(base_matrix_operator, LinearOperator), "The linear operator should be a LinearOperator object."

        super(ConjugateTransposeLinearOperator, self).__init__(base_matrix_operator.output_shape, base_matrix_operator.input_shape)

        self.base_matrix_operator = base_matrix_operator
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate_transpose(x)
    
    def conjugate(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.transpose(x)
    
    def transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.conjugate(y)
    
    def conjugate_transpose(self, y: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.forward(y)
    

class InverseLinearOperator(InvertibleLinearOperator):
    def __init__(self, base_matrix_operator: InvertibleLinearOperator):
        """
        This is an abstract class for linear operators that are the inverse of another linear operator.

        It inherits from SquareLinearOperator.

        parameters:
            base_matrix_operator: LinearOperator object
                The linear operator to which the inverse should be applied.
        """
        assert isinstance(base_matrix_operator, InvertibleLinearOperator), "The input linear operator should be a InvertibleLinearOperator object."

        super(InverseLinearOperator, self).__init__()

        self.base_matrix_operator = base_matrix_operator
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_matrix_operator.inverse(x)
    
    def inverse_LinearOperator(self):
        return self.base_matrix_operator.forward_LinearOperator()

class CompositeLinearOperator(LinearOperator):
    def __init__(self, matrix_operators):
        """
        This class represents the matrix-matrix product of multiple linear operators.

        It inherits from LinearOperator.

        parameters:
            operators: list of LinearOperator objects
                The list of linear operators to be composed. The product is taken in the order they are provided.
        """

        assert isinstance(matrix_operators, list), "The operators should be provided as a list of LinearOperator objects."
        assert len(matrix_operators) > 0, "At least one operator should be provided."
        for operator in matrix_operators:
            assert isinstance(operator, LinearOperator), "All operators should be LinearOperator objects."

        LinearOperator.__init__(self)

        self.matrix_operators = matrix_operators

    def forward(self, x):
        result = x
        for matrix_operator in self.matrix_operators:
            result = matrix_operator.forward(result)
        return result
    
    def transpose(self,y):
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.transpose(result)
        return result
    
    def inverse(self, y):
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.inverse(result)
        return result
    




class InvertibleCompositeLinearOperator(CompositeLinearOperator, InvertibleLinearOperator):
    def __init__(self, matrix_operators):
        """
        This class represents the matrix-matrix product of multiple linear operators.

        It inherits from LinearOperator.

        parameters:
            operators: list of LinearOperator objects
                The list of linear operators to be composed. The product is taken in the order they are provided.
        """
        for operator in matrix_operators:
            assert isinstance(operator, InvertibleLinearOperator), "All operators should be InvertibleLinearOperator objects."
        CompositeLinearOperator.__init__(self, matrix_operators)

    def inverse(self, y):
        result = y
        for matrix_operator in reversed(self.matrix_operators):
            result = matrix_operator.inverse(result)
        return result
    
    def inverse_LinearOperator(self):
        return CompositeLinearOperator([operator.inverse_LinearOperator() for operator in reversed(self.matrix_operators)])

class EigenDecomposedLinearOperator(CompositeLinearOperator, SymmetricLinearOperator):
    def __init__(self, eigenvalue_matrix: DiagonalLinearOperator, eigenvector_matrix: InvertibleLinearOperator):
        """
        This class represents a linear operator that is given by its eigenvalue decomposition.

        It inherits from CompositeLinearOperator and SymmetricLinearOperator

        parameters:
            eigenvalue_matrix: DiagonalLinearOperator object
                The diagonal matrix of eigenvalues.
            eigenvector_matrix: InvertibleLinearOperator object
                The invertible matrix of eigenvectors.
        """

        assert isinstance(eigenvalue_matrix, DiagonalLinearOperator), "The eigenvalues should be a DiagonalLinearOperator object."
        assert isinstance(eigenvector_matrix, InvertibleLinearOperator), "The eigenvectors should be a InvertibleLinearOperator object."

        self.eigenvalue_matrix = eigenvalue_matrix
        self.eigenvectors = eigenvector_matrix

        operators = [eigenvector_matrix, eigenvalue_matrix, eigenvector_matrix.inverse_LinearOperator()]

        CompositeLinearOperator.__init__(self, operators)







class SingularValueDecomposedLinearOperator(CompositeLinearOperator):
    def __init__(self, left_singular_vector_matrix: UnitaryLinearOperator, singular_value_matrix: DiagonalLinearOperator,  right_singular_vector_matrix: UnitaryLinearOperator):
        """
        This class represents a linear operator that is given by its singular value decomposition.

        It inherits from SquareLinearOperator.

        parameters:
            singular_values: DiagonalLinearOperator object
                The diagonal matrix of singular values.
            left_singular_vectors: UnitaryLinearOperator object
                The matrix of left singular vectors.
            right_singular_vectors: UnitaryLinearOperator object
                The matrix of right singular vectors.
        """

        assert isinstance(singular_value_matrix, DiagonalLinearOperator), "The singular values should be a DiagonalLinearOperator object."
        assert isinstance(left_singular_vector_matrix, UnitaryLinearOperator), "The left singular vectors should be a UnitaryLinearOperator object."
        assert isinstance(right_singular_vector_matrix, UnitaryLinearOperator), "The right singular vectors should be a UnitaryLinearOperator object."

        operators = [left_singular_vector_matrix, singular_value_matrix, right_singular_vector_matrix.conjugate_transpose_LinearOperator()]

        super(SingularValueDecomposedLinearOperator, self).__init__(operators)

        self.singular_values = singular_value_matrix
        self.left_singular_vectors = left_singular_vector_matrix
        self.right_singular_vectors = right_singular_vector_matrix

