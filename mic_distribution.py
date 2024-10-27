import torch
import torch.nn as nn
import torch.optim as optim
from diffusion_laboratory.sde import StochasticDifferentialEquation
from diffusion_laboratory.linalg import ScalarLinearOperator

from diffusion_laboratory.sde import ScalarSDE


import matplotlib.animation as animation

step_size = 0.5

def f(x, t): 
    return 0.5*step_size*step_size*mic.score(x)

def G(x, t): 
    return ScalarLinearOperator(step_size)


class MIC_Distribution:
    def __init__(self, bandwidth=0.1):
        self.bandwidth = bandwidth  # Standard deviation for isotropic Gaussians
        self.points_M = self.create_M()
        self.points_I = self.create_I()
        self.points_C = self.create_C()
        
        
  
        self.points_all = torch.concat([self.points_M,self.points_I,self.points_C],dim=0)
        self.points_M -= torch.mean(self.points_all, dim=0, keepdim=True)
        self.points_I -= torch.mean(self.points_all, dim=0, keepdim=True)  
        self.points_C -= torch.mean(self.points_all, dim=0, keepdim=True)
        self.points_all -= torch.mean(self.points_all, dim=0, keepdim=True)
        
        self.points_M *= 3.0 # Scale the points to have a larger spread
        self.points_I *= 3.0 # Scale the points to have a larger spread
        self.points_C *= 3.0 # Scale the points to have a larger spread
        self.points_all *= 3.0 # Scale the points to have a larger spread
        
        
        self.pointset=[self.points_M,self.points_I,self.points_C]
        
        
        self.num_points =self.points_M.size(0)+self.points_I.size(0)+ self.points_C.size(0)
    
    def create_M(self):
        """Creates evenly spaced points for the letter 'M'."""
        left_line = torch.stack([torch.linspace(0, 0, 10), torch.linspace(0, 1, 10)], dim=1)
        right_line = torch.stack([torch.linspace(1, 1, 10), torch.linspace(0, 1, 10)], dim=1)
        diag1 = torch.stack([torch.linspace(0, 0.5, 8), torch.linspace(1, 0.5, 8)], dim=1)
        diag2 = torch.stack([torch.linspace(0.5, 1, 8), torch.linspace(0.5, 1, 8)], dim=1)
        return torch.cat([left_line, right_line, diag1, diag2], dim=0)

    def create_I(self):
        """Creates evenly spaced points for the letter 'I'."""
        vertical_line = torch.stack([torch.linspace(1.5, 1.5, 10), torch.linspace(0, 1, 10)], dim=1)
        top_line = torch.stack([torch.linspace(1.4, 1.6, 3), torch.ones(3)], dim=1)
        bottom_line = torch.stack([torch.linspace(1.4, 1.6, 3), torch.zeros(3)], dim=1)
        return torch.cat([vertical_line, top_line, bottom_line], dim=0)

  


    def create_C(self):
        """Creates evenly spaced points for the letter 'C'."""
        theta = torch.linspace(torch.pi/2, torch.pi*3/2, 14)
        x = 2.5 + 0.5*torch.cos(theta)  # Offset in the x-axis for the "C"
        y = 0.5 + 0.5*torch.sin(theta)
        return torch.stack([x, y], dim=1)
    
    def sample(self, char_id, n_samples=100):
        """Samples n points from the Gaussian mixture model."""
        # Sample random indices for the points
        indices = torch.randint(0, self.pointset[char_id].size(0), (n_samples,))
        chosen_points = self.pointset[char_id][indices]

        # Sample isotropic Gaussians around the chosen points
        samples = chosen_points + torch.randn(n_samples, 2) * self.bandwidth
        return samples

    def log_pdf(self, x):
        """Computes the log probability density function of the Gaussian mixture at points x."""
        x = x.unsqueeze(1)  # [n_samples, 1, 2]
        diffs = x - self.points_all.unsqueeze(0)  # [n_samples, num_points, 2]
        exponent = -0.5 * torch.sum(diffs ** 2, dim=2) / (self.bandwidth ** 2)
        log_probs = exponent - torch.log(2 * torch.pi * self.bandwidth ** 2)
        return torch.logsumexp(log_probs, dim=1) - torch.log(torch.tensor(self.num_points))

    def score(self, x):
        """Computes the score, which is the gradient of the log probability density function."""
        x = x.unsqueeze(1)  # [n_samples, 1, 2]
        diffs = x - self.points_all.unsqueeze(0)  # [n_samples, num_points, 2]
        exponent = -0.5 * torch.sum(diffs ** 2, dim=2) / (self.bandwidth ** 2)
        log_probs = exponent - torch.log(torch.tensor(2 * torch.pi * self.bandwidth ** 2))
        weights = torch.softmax(log_probs, dim=1)  # Softmax to get weights
        score = torch.sum(weights.unsqueeze(2) * (-diffs / (self.bandwidth ** 2)), dim=1)
        return score


    
if __name__ == "__main__":
    
    
    import matplotlib.pyplot as plt

    # Create the MIC_Distribution instance
    mic = MIC_Distribution(bandwidth=0.1)
    n=1000
    # Sample 1000 points
    samples_M = mic.sample(0,n)
    samples_I = mic.sample(1,n)
    samples_C = mic.sample(2,n)
    samples =torch.concat([samples_M,samples_I,samples_C],dim=0)
    
    # Plot the sampled points
    plt.figure(figsize=(6, 6))
    plt.title("2D Gaussian Mixture Model Sampling for 'MIC'")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.scatter(samples[:n, 0].numpy(), samples[:n, 1].numpy(), s=2, color='red', alpha=0.6)
    plt.scatter(samples[n:n*2, 0].numpy(), samples[n:n*2, 1].numpy(), s=2, color='blue', alpha=0.6)
    plt.scatter(samples[n*2:n*3:, 0].numpy(), samples[n*2:n*3, 1].numpy(), s=2, color='green', alpha=0.6)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("mic_distribution.png")

        
        
        
    langevin_dynamics_SDE = StochasticDifferentialEquation(f=f, G=G)

    timesteps = torch.linspace(0.0, 1.0, 100)
    xt = langevin_dynamics_SDE.sample(samples, timesteps, sampler='euler', return_all=True, verbose=True)

    # plot the samples
    fig = plt.figure(figsize=(6, 6))
    ln_M, = plt.plot(xt[0][:n, 0], xt[0][:n, 1], 'o', color='red', markersize=1)
    ln_I, = plt.plot(xt[0][n:n*2, 0], xt[0][n:n*2, 1], 'o', color='blue', markersize=1)
    ln_C, = plt.plot(xt[0][n*2:n*3, 0], xt[0][n*2:n*3, 1], 'o', color='green', markersize=1)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)


    # Add labels
    plt.legend(loc="upper right")

    # Update function for animation
    def update(frame):
        ln_M.set_data(xt[frame][:n, 0], xt[frame][:n, 1])
        ln_I.set_data(xt[frame][n:n*2, 0], xt[frame][n:n*2, 1])
        ln_C.set_data(xt[frame][n*2:n*3, 0], xt[frame][n*2:n*3, 1])
        print(f"Animated Frame: {frame}")
        return ln_M, ln_I, ln_C

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(xt), blit=False)

    # Save the animation using ffmpeg
    writer = animation.FFMpegWriter(fps=10, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
    ani.save('mic_langevin_dynamics.mp4', writer=writer, dpi=300)

    print("Done!")
    
    
    
        
    beta = torch.tensor(3.0)
    signal_scale = lambda t: torch.exp(-0.5*beta*t)
    noise_variance = lambda t: (1 - torch.exp(-beta*t))
    signal_scale_prime = lambda t: -0.5*beta*torch.exp(-0.5*beta*t)
    noise_variance_prime = lambda t: beta*torch.exp(-beta*t)

    forward_diffusion_SDE = ScalarSDE(  signal_scale=signal_scale,
                                        noise_variance=noise_variance,
                                        signal_scale_prime=signal_scale_prime,
                                        noise_variance_prime=noise_variance_prime)

    timesteps = torch.linspace(0.0, 1.0, 100)
    xt = forward_diffusion_SDE.sample(samples, timesteps, sampler='euler', return_all=True, verbose=True)


    # plot the samples
    fig = plt.figure(figsize=(6, 6))
    ln_M, = plt.plot(xt[0][:n, 0], xt[0][:n, 1], 'o', color='red', markersize=1)
    ln_I, = plt.plot(xt[0][n:n*2, 0], xt[0][n:n*2, 1], 'o', color='blue', markersize=1)
    ln_C, = plt.plot(xt[0][n*2:n*3, 0], xt[0][n*2:n*3, 1], 'o', color='green', markersize=1)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)


    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(xt), blit=False)

    # Higher quality video with ffmpeg settings
    writer = animation.writers['ffmpeg'](fps=10, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
    ani.save('mic_forward_diffusion.mp4', writer=writer, dpi=300)

    print("Done!")
    
        
        
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timesteps = torch.linspace(0.0, 1.0, 100)

    # Define the score estimator network (3 -> 2 dense NN)
    class Denoiser(nn.Module):
        def __init__(self):
            super(Denoiser, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(3, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Linear(64, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 512),
                nn.LayerNorm(512),
                nn.SiLU(),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.SiLU(),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.SiLU(),
                nn.Linear(64, 2)  # Output 2D score estimate
            )
        
        def forward(self, x, t):
            # Concatenate 2D position and time
            input = torch.cat([x, t], dim=1)
            return self.fc(input)

    # Instantiate the score estimator
    denoiser = Denoiser().to(device)

    # Adam optimizer
    optimizer = optim.Adam(denoiser.parameters(), lr=1e-3)

    T = 0.5
    # Training loop for score matching
    def train_score_estimator(forward_sde, score_estimator, num_epochs=1000, batch_size=1000):
        active_batch=batch_size*3 # M, I, C
        
        for epoch in range(num_epochs):
            # Sample 1000 points
            samples_M = mic.sample(0,1000)
            samples_I = mic.sample(1,1000)
            samples_C = mic.sample(2,1000)
            x0 =torch.concat([samples_M,samples_I,samples_C],dim=0).to(device)  # Sample from the MIC distribution
            
            
            
            #x0 = mic.sample(active_batch).to(device)  # Sample from the MIC distribution
            t = T*torch.rand(active_batch, 1).to(device)  # Uniformly sample t from [0, 1)
            t = t**2.0 # square it to emphasize early time points

            # Get x_t given x_0 from the forward SDE
            xt = forward_sde.sample_x_t_given_x_0(x0, t)
            
            # Compute the score estimate
            x0_est = score_estimator(xt, t)
            
            # Compute the true score (ground truth)
            # Sigma = forward_sde.Sigma(t)
            # Sigma.scalar += 1e-6  # Add small constant to avoid division by zero
            # mu = forward_sde.H(t) @ x0
            # target_score = -1.0 * (Sigma.inverse_LinearOperator() @ (xt - mu))

            # Score matching loss (MSE between estimated and true score)
            loss = ((x0_est - x0) ** 2).mean()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")


        # save the trained score estimator
        torch.save(denoiser.state_dict(), 'denoiser.pth')
    # if the weights exist, load them
    try:
        denoiser.load_state_dict(torch.load('denoiser.pth'))
    except:
        print("No weights found for the score estimator. Training the score estimator...")

    # Train the score estimator
    train_score_estimator(forward_diffusion_SDE, denoiser)



    def score_estimator(x, t):
        x0_pred = denoiser(x, t)
        Sigma = forward_diffusion_SDE.Sigma(t)
        Sigma.scalar += 1e-8  # Add small constant to avoid division by zero
        mu = forward_diffusion_SDE.H(t) @ x0_pred
        return -1.0 * (Sigma.inverse_LinearOperator() @ (x - mu))


    # Get the reverse SDE using the trained score estimator
    reverse_diffusion_SDE = forward_diffusion_SDE.reverse_SDE_given_score_estimator(score_estimator)

    # Sample points using the reverse SDE
    # xT = torch.randn(1000, 2).to(device)
    # x0 = mic.sample(1000)
    # xT = forward_diffusion_SDE.sample_x_t_given_x_0(x0, torch.tensor(1.0).to(device))
    
  
    
    xT=xt[-1].to(device)


    
    #xT = torch.randn(1000, 2).to(device)

    reverse_timesteps = torch.linspace(T, 0.0, 100).to(device)
    denoiser.eval()
    xt_reverse = reverse_diffusion_SDE.sample(xT, reverse_timesteps, sampler='euler', return_all=True, verbose=True)

    
    # plot the samples
    fig = plt.figure(figsize=(6, 6))
    ln_M, = plt.plot(xt_reverse[0][0:n, 0], xt_reverse[0][0:n, 1], 'o', color='red', markersize=1)
    ln_I, = plt.plot(xt_reverse[0][n:n*2, 0], xt_reverse[0][n:n*2, 1], 'o', color='blue', markersize=1)
    ln_C, = plt.plot(xt_reverse[0][n*2:n*3, 0], xt_reverse[0][n*2:n*3, 1], 'o', color='green', markersize=1)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    # Update function for animation
    def update(frame):
        ln_M.set_data(xt_reverse[frame][0:n, 0].detach(), xt_reverse[frame][0:n, 1].detach())
        ln_I.set_data(xt_reverse[frame][n:n*2, 0].detach(), xt_reverse[frame][n:n*2, 1].detach())
        ln_C.set_data(xt_reverse[frame][n*2:n*3, 0].detach(), xt_reverse[frame][n*2:n*3, 1].detach())
        print(f"Animated Frame: {frame}")
        return ln_M, ln_I, ln_C


    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(xt_reverse), blit=False)

    # Higher quality video with ffmpeg settings
    writer = animation.writers['ffmpeg'](fps=10, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
    ani.save('mic_reverse_diffusion.mp4', writer=writer, dpi=300)

    print("Done!")
