# random_tensor_laboratory/diffusion/sde.py

import torch
import torch.nn as nn
import torch
from ..linalg import LinearOperator
from ..linalg import InvertibleLinearOperator, SymmetricLinearOperator
from ..linalg import ScalarLinearOperator
import numpy as np

class StochasticDifferentialEquation(nn.Module):
    def __init__(self, f, G):
        """
        This class implements an Ito stochastic differential equation (SDE) of the form 
        
        dx = f(x, t) dt + G(x, t) dw
        
        f is a vector-valued function of x and t representing the drift term 
        and G is a matrix-valued function of x and t representing the diffusion rate.

        parameters:
            f: callable
                The drift term of the SDE. It should take x and t as input and return a tensor of the same shape as x.
            G: callable
                The diffusion term of the SDE. It should take x and t as input and return a rtl.linalg.LinearOperator that can act on a tensor of the same shape as x.
        """

        super(StochasticDifferentialEquation, self).__init__()

        self.f = f
        self.G = G

    def forward(self, x, t):
        return self.f(x, t), self.G(x, t)
    
    def reverse_SDE_given_score_estimator(self, score_estimator):
        """
        This method returns the time reversed StochasticDifferentialEquation given a score function estimator.

        The time reversed SDE is given by

        dx = f*(x, t) dt + G(x, t) dw

        where f*(x, t) = f(x, t) - div_x( G(x,t) G(x,t)^T ) - G(x, t) G(x, t)^T score_estimator(x, t)

        parameters:
            score_estimator: callable
                The score estimator function that takes x, t, as input and returns the score function estimate.
        returns:
            sde: StochasticDifferentialEquation
                The time reversed SDE.
        """
        _f = self.f
        _G = self.G

        def compute_divergence_fn(GG_T, x):
            x_flattened = x.view(-1)  # Flatten the x tensor
            div = torch.zeros_like(x_flattened)
            for i in range(x_flattened.shape[0]):
                unit_vector = torch.zeros_like(x_flattened)
                unit_vector[i] = 1.0
                GG_T_unit = GG_T(unit_vector.view_as(x)).view(-1)
                try:
                    grad = torch.autograd.grad(GG_T_unit.sum(), x, retain_graph=True, create_graph=True)[0]
                    div[i] = grad.view(-1)[i]
                except RuntimeError:
                    # If gradient computation fails, set divergence to zero
                    div[i] = 0.0
            return div.view_as(x)  # Reshape the divergence to the original shape of x

        def _f_star(x, t):
            G_t = _G(x, t)
            G_tT = G_t.transpose_LinearOperator()
            GG_T = lambda v: G_t(G_tT(v))  # Define GG_T as a function to apply G_t and its transpose

            div_GG_T = compute_divergence_fn(GG_T, x)
            return _f(x, t) - div_GG_T - GG_T(score_estimator(x, t))

        return StochasticDifferentialEquation(f=_f_star, G=_G)

    def sample(self, x, timesteps, sampler='euler', return_all=False, verbose=False):
        """
        This method samples from the SDE.

        parameters:
            x: torch.Tensor
                The initial condition.
            timesteps: torch.Tensor
                The time steps at which the SDE is evaluated.
            sampler: str
                The method used to compute the forward update. Currently, only 'euler' and 'heun' are supported.
        returns:
            x: torch.Tensor
                The output tensor.
        """

        x = x
        # t = timesteps[0]

        # if timesteps is not a tensor, make it a tensor
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps)
    
        t = timesteps[0].reshape(1,1).repeat(x.shape[0], 1)

        if return_all:
            x_all = [x]
        
        for i in range(1, len(timesteps)):
            if verbose:
                print(f"Sampling step {i}/{len(timesteps)-1}")
            last_step = i == len(timesteps) - 1
            dt = timesteps[i] - t
            x = self._sample_step(x, t, dt, sampler=sampler, last_step=last_step)
            t = timesteps[i].reshape(1,1).repeat(x.shape[0], 1)

            if return_all:
                x_all.append(x)
        
        if return_all:
            return x_all
        
        return x

    def _sample_step(self, x, t, dt, sampler='euler', last_step=False):
        """
        This method computes the forward update of the SDE.

        The forward SDE is given by

        dx = f(x, t) dt + G(x, t) dw

        parameters:
            x: torch.Tensor
                The input tensor.
            t: float
                The time at which the SDE is evaluated.
            dt: float or torch.Tensor
                The time step.
            sampler: str
                The method used to compute the forward update. Currently, 'euler' and 'heun' are supported.
        returns:
            dx: torch.Tensor
                The output tensor.
        """

        if sampler == 'euler':
            return self._sample_step_euler(x, t, dt, last_step=last_step)
        elif sampler == 'heun':
            return self._sample_step_heun(x, t, dt, last_step=last_step)
        else:
            raise ValueError("The sampler should be one of ['euler', 'heun'].")

    def _sample_step_euler(self, x, t, dt, last_step=False):
        """
        This method computes the forward update of the SDE using the Euler-Maruyama method.

        The forward SDE is given by

        dx = f(x, t) dt + G(x, t) dw

        parameters:
            x: torch.Tensor
                The input tensor.
            t: float
                The time at which the SDE is evaluated.
            dt: float or torch.Tensor
                The time step.
            dw: torch.Tensor
                The Wiener process increment.
        returns:
            dx: torch.Tensor
                The output tensor.
        """

        if isinstance(dt, float):
            dt = torch.tensor(dt)

        dw = torch.randn_like(x) * torch.sqrt(torch.abs(dt))

        _f = self.f(x, t)
        assert isinstance(_f, torch.Tensor), "The drift term f(x, t) should return a tensor."
        assert _f.shape == x.shape, "The drift term f(x, t) should return a tensor of the same shape as x."
        
        _G = self.G(x, t)
        assert isinstance(_G, LinearOperator), "The diffusion term G(x, t) should return a LinearOperator."
        
        _f_dt = _f * dt
        _G_dw = _G @ dw

        if last_step:
            return x + _f_dt
        else:
            return x + _f_dt + _G_dw

    def _sample_step_heun(self, x, t, dt, last_step=False):
        """
        This method computes the forward update of the SDE using the Heun's method.

        The forward SDE is given by

        dx = f(x, t) dt + G(x, t) dw

        parameters:
            x: torch.Tensor
                The input tensor.
            t: float
                The time at which the SDE is evaluated.
            dt: float or torch.Tensor
                The time step.
        returns:
            dx: torch.Tensor
                The output tensor.
        """

        if isinstance(dt, float):
            dt = torch.tensor(dt)

        dw = torch.randn_like(x) * torch.sqrt(dt)

        # Predictor step using Euler-Maruyama
        f_t = self.f(x, t)
        G_t = self.G(x, t)
        x_predict = x + f_t * dt + G_t @ dw

        # Corrector step
        f_t_corrector = self.f(x_predict, t + dt)
        G_t_corrector = self.G(x_predict, t + dt)

        f_avg = (f_t + f_t_corrector) / 2
        G_dw_avg = (G_t @ dw + G_t_corrector @ dw) / 2


        if last_step:
            return x + f_avg * dt
        else:
            return x + f_avg * dt + G_dw_avg
    


class LinearSDE(StochasticDifferentialEquation):
    def __init__(self, H, Sigma, H_prime=None, Sigma_prime=None, F=None, G=None):
        """
        This class implements a linear stochastic differential equation (SDE) of the form:
        dx = F(t) @ x dt + G(t) dw
        where F and G are derived from H and Sigma if not directly provided.

        Parameters:
            H: callable
                Function that returns an InvertibleLinearOperator representing the system response.
            Sigma: callable
                Function that returns a SymmetricLinearOperator representing the covariance.
            H_prime: callable, optional
                Function that returns the time derivative of H. If not provided, it will be computed automatically.
            Sigma_prime: callable, optional
                Function that returns the time derivative of Sigma. If not provided, it will be computed automatically.
            F: callable, optional
                Function that returns a LinearOperator representing the drift term. If not provided, it will be computed from H_prime and H.
            G: callable, optional
                Function that returns a LinearOperator representing the diffusion term. If not provided, it will be computed from Sigma_prime, F, and Sigma.

        Requirements:
            - H must return an InvertibleLinearOperator.
            - Sigma must return a SymmetricLinearOperator.
            - The @ operator must be implemented for matrix-matrix multiplication of F, Sigma, and their transposes.
            - The addition, subtraction, and sqrt_LinearOperator methods must be implemented for the resulting matrix operations on Sigma_prime and others.

        If H_prime and Sigma_prime are not provided, they will be computed using automatic differentiation.
        """

        assert isinstance(H(0), InvertibleLinearOperator), "H(t) must return an InvertibleLinearOperator."
        assert isinstance(Sigma(0), SymmetricLinearOperator), "Sigma(t) must return a SymmetricLinearOperator."

        self.H = H
        self.Sigma = Sigma
        self.H_prime = H_prime
        self.Sigma_prime = Sigma_prime
        self.F = F
        self._G = G

        assert H_prime is not None or F is not None, "Either H_prime or F must be provided."
        assert Sigma_prime is not None or G is not None, "Either Sigma_prime or G must be provided."

        if F is None and H_prime is not None:
            self.F = lambda t: self.H_prime(t) @ self.H(t).inverse_LinearOperator()

        if self._G is None and Sigma_prime is not None:
            self._G = lambda t: (self.Sigma_prime(t) - self.F(t) @ self.Sigma(t) - self.Sigma(t) @ self.F(t).transpose_LinearOperator()).sqrt_LinearOperator()

        _f = lambda x, t: self.F(t) @ x
        _G = lambda x, t: self._G(t)

        super(LinearSDE, self).__init__(f=_f, G=_G)
        
    def reverse_SDE_given_score_estimator(self, score_estimator):
        """
        This method returns the time reversed StochasticDifferentialEquation given a score function estimator.

        The time reversed SDE is given by

        dx = f*(x, t) dt + G(x, t) dw

        where f*(x, t) = f(x, t) - div_x( G(x,t) G(x,t)^T ) - G(x, t) G(x, t)^T score_estimator(x, t)

        parameters:
            score_estimator: callable
                The score estimator function that takes x, t, as input and returns the score function estimate.
        returns:
            sde: StochasticDifferentialEquation
                The time reversed SDE.
        """
        _f = self.f
        _G = self.G

        def compute_divergence(GG_T, x):
            # divergence always zero for operators that do not depend on x
            div = torch.zeros_like(x)
            return div  

        def _f_star(x, t):
            G_t = _G(x, t)
            G_tT = G_t.transpose_LinearOperator()
            GG_T = lambda v: G_t(G_tT(v))  # Define GG_T as a function to apply G_t and its transpose

            div_GG_T = compute_divergence(GG_T, x)
            return _f(x, t) - div_GG_T - GG_T(score_estimator(x, t))

        return StochasticDifferentialEquation(f=_f_star, G=_G)
    
    def reverse_SDE_given_mean_estimator(self, mean_estimator):
        """
        This method returns the time reversed StochasticDifferentialEquation given a mean function estimator.

        The time reversed SDE is given by

        dx = f*(x, t) dt + G(x, t) dw

        where f*(x, t) = f(x, t) - G(x, t) G(x, t)^T score_estimator(x, t)

        parameters:
            mean_estimator: callable
                The mean estimator function that takes x, t, as input and returns the mean function estimate.
        returns:
            sde: StochasticDifferentialEquation
                The time reversed SDE.
        """
        
        def score_estimator(x, t):
            mu_t = mean_estimator(x, t)
            sigma_t_inv = self.Sigma(t).inverse_LinearOperator()
            return sigma_t_inv @ (mu_t-x)

        return self.reverse_SDE_given_score_estimator(score_estimator)




    def reverse_SDE_given_noise_estimator(self, noise_estimator):
        """
        This method returns the time reversed StochasticDifferentialEquation given a noise function estimator.

        The time reversed SDE is given by

        dx = f*(x, t) dt + G(x, t) dw

        where f*(x, t) = f(x, t) - G(x, t) G(x, t)^T score_estimator(x, t)

        parameters:
            noise_estimator: callable
                The noise estimator function that takes x, t, as input and returns the noise function estimate.
        returns:
            sde: StochasticDifferentialEquation
                The time reversed SDE.
        """

        # score = Sigma(t)^(-1) @ (x - mu_t)
        # x = mu_t + Sigma(t)^(1/2) @ noise
        # noise = Sigma(t)^(-1/2) @ (x - mu_t)
        # score = Sigma(t)^(-1) @ (mu_t + Sigma(t)^(1/2) @ noise - mu_t)
        # score = Sigma(t)^(-1) @ Sigma(t)^(1/2) @ noise
        # score = Sigma(t)^(-1/2) @ noise
        
        def score_estimator(x, t):
            noise_t = noise_estimator(x, t)
            sigma_t_sqrt_inv = self.Sigma(t).sqrt_LinearOperator().inverse_LinearOperator()
            return -1.0*(sigma_t_sqrt_inv @ noise_t)

        return self.reverse_SDE_given_score_estimator(score_estimator)
    
    def mean_response_x_t_given_x_0(self, x0, t):
        """
        Computes the mean response of x_t given x_0.

        Parameters:
            x0: torch.Tensor
                The initial condition.
            t: float
                The time at which the mean response is evaluated.
        
        Returns:
            torch.Tensor
                The mean response at time t.
        """
        return self.H(t) @ x0

    def sample_x_t_given_x_0(self, x0, t):
        """
        Samples x_t given x_0 using the mean response and adding Gaussian noise with covariance Sigma(t).

        Parameters:
            x0: torch.Tensor
                The initial condition.
            t: float
                The time at which the sample is evaluated.
        
        Returns:
            torch.Tensor
                The sampled response at time t.
        """
        noise = torch.randn_like(x0)
        return self.sample_x_t_given_x_0_and_noise(x0, noise, t)

        
    

    def sample_x_t_given_x_0_and_noise(self, x0, noise, t):
        """
        Samples x_t given x_0 using the mean response and adding Gaussian noise with covariance Sigma(t).

        Parameters:
            x0: torch.Tensor
                The initial condition.
            t: float
                The time at which the sample is evaluated.
        
        Returns:
            torch.Tensor
                The sampled response at time t.
        """
        mean_response = self.mean_response_x_t_given_x_0(x0, t)
        Sigma_sqrtm = self.Sigma(t).sqrt_LinearOperator()
        return mean_response + Sigma_sqrtm @ noise 

    def reverse_SDE_given_posterior_mean_estimator(self, posterior_mean_estimator):
        """
        Constructs the reverse-time stochastic differential equation given a posterior mean estimator.

        The time-reversed SDE is given by:
        dx = f*(x, t) dt + G(x, t) dw
        where f*(x, t) = f(x, t) - G(x, t) G(x, t)^T score_estimator(x, t)
        and score_estimator(x, t) = Sigma(t)^(-1) @ (x - mu_t)

        Parameters:
            posterior_mean_estimator: callable
                Function that takes x and t as input and returns the estimated mean at time t.
        
        Returns:
            StochasticDifferentialEquation
                The reverse-time SDE.
        """
        
        def score_estimator(x, t):
            mu_t = posterior_mean_estimator(x, t)
            sigma_t_inv = self.Sigma(t).inverse_LinearOperator()
            return sigma_t_inv @ (x - mu_t)

        return self.reverse_SDE_given_score_estimator(score_estimator)

        
class ScalarSDE(LinearSDE):
    def __init__(self, signal_scale, noise_variance, signal_scale_prime=None, noise_variance_prime=None):
        """
        This class implements a scalar homogeneous stochastic differential equation (SDE).

        dx = f(t) x dt + g(t) dw

        Parameters:
            signal_scale: callable
                Function that returns a scalar representing the system response.
            noise_variance: callable
                Function that returns a scalar representing the covariance.
            signal_scale_prime: callable, optional
                Function that returns the time derivative of signal_scale. If not provided, it will be computed automatically.
            noise_variance_prime: callable, optional
                Function that returns the time derivative of noise_variance. If not provided, it will be computed automatically.
        """
        assert isinstance(signal_scale(0), (float, torch.Tensor)), "signal_scale(t) must return a scalar."
        assert isinstance(noise_variance(0), (float, torch.Tensor)), "noise_variance(t) must return a scalar."

        H = lambda t: ScalarLinearOperator(signal_scale(t))
        Sigma = lambda t: ScalarLinearOperator(noise_variance(t))

        if signal_scale_prime is None:
            signal_scale_prime = lambda t: torch.autograd.grad(signal_scale(t), t, create_graph=True)[0]
        if noise_variance_prime is None:
            noise_variance_prime = lambda t: torch.autograd.grad(noise_variance(t), t, create_graph=True)[0]

        H_prime = lambda t: ScalarLinearOperator(signal_scale_prime(t))
        Sigma_prime = lambda t: ScalarLinearOperator(noise_variance_prime(t))

        super(ScalarSDE, self).__init__(H, Sigma, H_prime, Sigma_prime)

