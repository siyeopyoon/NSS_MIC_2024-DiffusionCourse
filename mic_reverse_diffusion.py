import torch
import torch.nn as nn
import torch.optim as optim
from diffusion_laboratory.sde import StochasticDifferentialEquation
from diffusion_laboratory.linalg import ScalarLinearOperator

from diffusion_laboratory.sde import ScalarSDE


import matplotlib.animation as animation






class CAMCA_Distribution:
    def __init__(self, bandwidth=0.1):
        self.bandwidth = bandwidth  # Standard deviation for isotropic Gaussians
        self.points_C1 = self.create_C()
        self.points_A1 = self.create_A()
        self.points_M = self.create_M()
        self.points_C2 = self.create_C()
        self.points_A2 = self.create_A()

        
        self.points_C1 -= torch.mean(self.points_C1, dim=0, keepdim=True) +torch.tensor((-1.2,0.0))
        self.points_A1 -= torch.mean(self.points_A1, dim=0, keepdim=True) +torch.tensor((-2.1,0.0))  
        self.points_M -= torch.mean(self.points_M, dim=0, keepdim=True)
        self.points_C2 -= torch.mean(self.points_C2, dim=0, keepdim=True) +torch.tensor((2.1,0.0))  
        self.points_A2 -= torch.mean(self.points_A2, dim=0, keepdim=True) +torch.tensor((1.2,0.0))  
        

        self.points_all = torch.concat([self.points_C1,self.points_A1,self.points_M,self.points_C2,self.points_A2],dim=0)
        
        self.points_C1 -= torch.mean(self.points_all, dim=0, keepdim=True)
        self.points_A1 -= torch.mean(self.points_all, dim=0, keepdim=True)  
        self.points_M -= torch.mean(self.points_all, dim=0, keepdim=True)
        self.points_C2 -= torch.mean(self.points_all, dim=0, keepdim=True)
        self.points_A2 -= torch.mean(self.points_all, dim=0, keepdim=True) 
        self.points_all -= torch.mean(self.points_all, dim=0, keepdim=True)
        
        
        
        self.points_C1 *= 3.0 # Scale the points to have a larger spread
        self.points_A1 *= 3.0 # Scale the points to have a larger spread
        self.points_M *= 3.0 # Scale the points to have a larger spread
        self.points_C2 *= 3.0 # Scale the points to have a larger spread
        self.points_A2 *= 3.0 # Scale the points to have a larger spread
        self.points_all *= 3.0 # Scale the points to have a larger spread

        self.pointset=[self.points_C1,self.points_A1,self.points_M,self.points_C2,self.points_A2]
        
        
    
    
    def create_A(self):
        """Creates evenly spaced points for the letter 'A'."""
        
        diag1 = torch.stack([torch.linspace(0, 0.2, 8), torch.linspace(0.2, 1, 8)], dim=1)
        diag2 = torch.stack([torch.linspace(0.2, 1, 8), torch.linspace(1, 0.2 ,8)], dim=1)
        
        horiz = torch.stack([torch.linspace(0.0, 0.5, 8), torch.linspace(0.5, 0.5, 8)], dim=1)
        
        
        return torch.cat([diag1, diag2,horiz], dim=0)
    
    
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

    # Create the camca_Distribution instance
    camca_dist = CAMCA_Distribution(bandwidth=0.1)
    n=1000
    # Sample 1000 points
    samples_C1= camca_dist.sample(0,n)
    samples_A1 = camca_dist.sample(1,n)
    samples_M = camca_dist.sample(2,n)
    samples_C2= camca_dist.sample(3,n)
    samples_A2 = camca_dist.sample(4,n)
    CAMCA=[samples_C1,samples_A1,samples_M,samples_C2,samples_A2]
    samples =torch.concat(CAMCA,dim=0)
    
    # Plot the sampled points
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[0:n, 0].numpy(), samples[0:n, 1].numpy(), s=2, color='red', alpha=0.6)
    plt.scatter(samples[n:n*2, 0].numpy(), samples[n:n*2, 1].numpy(), s=2, color='blue', alpha=0.6)
    plt.scatter(samples[n*2:n*3, 0].numpy(), samples[n*2:n*3, 1].numpy(), s=2, color='green', alpha=0.6)
    plt.scatter(samples[n*3:n*4, 0].numpy(), samples[n*3:n*4:, 1].numpy(), s=2, color='purple', alpha=0.6)
    plt.scatter(samples[n*4:n*5, 0].numpy(), samples[n*4:n*5:, 1].numpy(), s=2, color='orange', alpha=0.6)
    
    plt.title("2D Gaussian Mixture Model Sampling for 'CAMCA'")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("camca_distribution.png")


    def f(x, t): 
        return camca_dist.score(x)
    def G(x, t): 
        return ScalarLinearOperator(1.0)

        
    langevin_dynamics_SDE = StochasticDifferentialEquation(f=f, G=G)
    timesteps = torch.linspace(0.0, 1.0, 100)
    xt = langevin_dynamics_SDE.sample(samples, timesteps, sampler='euler', return_all=True, verbose=True)

    # plot the samples
    fig = plt.figure(figsize=(6, 6))
    ln_C1, = plt.plot(xt[0][:n, 0], xt[0][:n, 1], 'o', color='red', markersize=1)
    ln_A1, = plt.plot(xt[0][n:n*2, 0], xt[0][n:n*2, 1], 'o', color='blue', markersize=1)
    ln_M,  = plt.plot(xt[0][n*2:n*3, 0], xt[0][n*2:n*3, 1], 'o', color='green', markersize=1)
    ln_C2, = plt.plot(xt[0][n*3:n*4, 0], xt[0][n*3:n*4, 1], 'o', color='purple', markersize=1)
    ln_A2, = plt.plot(xt[0][n*4:n*5, 0], xt[0][n*4:n*5, 1], 'o', color='orange', markersize=1)
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)

    # Add labels
    plt.legend(loc="upper right")

    # Update function for animation
    def update(frame):
        ln_C1.set_data(xt[frame][:n, 0], xt[frame][:n, 1])
        ln_A1.set_data(xt[frame][n:n*2, 0], xt[frame][n:n*2, 1])
        ln_M.set_data(xt[frame][n*2:n*3, 0], xt[frame][n*2:n*3, 1])
        ln_C2.set_data(xt[frame][n*3:n*4, 0], xt[frame][n*3:n*4, 1])
        ln_A2.set_data(xt[frame][n*4:n*5, 0], xt[frame][n*4:n*5, 1])
        #print(f"Animated Frame: {frame}")
        return ln_C1, ln_A1, ln_M,ln_C2,ln_A2

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(timesteps), blit=False)

    # Save the animation using ffmpeg
    writer = animation.FFMpegWriter(fps=10, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
    ani.save('camca_langevin_dynamics.mp4', writer=writer, dpi=300)

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
    ln_C1, = plt.plot(xt[0][:n, 0], xt[0][:n, 1], 'o', color='red', markersize=1)
    ln_A1, = plt.plot(xt[0][n:n*2, 0], xt[0][n:n*2, 1], 'o', color='blue', markersize=1)
    ln_M, = plt.plot(xt[0][n*2:n*3, 0], xt[0][n*2:n*3, 1], 'o', color='green', markersize=1)
    ln_C2, = plt.plot(xt[0][n*3:n*4, 0], xt[0][n*3:n*4, 1], 'o', color='purple', markersize=1)
    ln_A2, = plt.plot(xt[0][n*4:n*5, 0], xt[0][n*4:n*5, 1], 'o', color='orange', markersize=1)
  
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)



    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(timesteps), blit=False)

    # Higher quality video with ffmpeg settings
    writer = animation.writers['ffmpeg'](fps=10, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
    ani.save('camca_forward_diffusion.mp4', writer=writer, dpi=300)

    print("Done!")
    
        
        
        
    # Define the score estimator network (2d points + time  -> denoised 2d points)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class Denoiser(nn.Module):
        def __init__(self):
            super(Denoiser, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(3, 64),   nn.LayerNorm(64), nn.SiLU(),
                nn.Linear(64, 128), nn.LayerNorm(128),nn.SiLU(),
                nn.Linear(128, 256),nn.LayerNorm(256),nn.SiLU(),
                nn.Linear(256, 512),nn.LayerNorm(512),nn.SiLU(),
                nn.Linear(512, 256),nn.LayerNorm(256),nn.SiLU(),
                nn.Linear(256, 128),nn.LayerNorm(128),nn.SiLU(),
                nn.Linear(128, 256),nn.LayerNorm(256),nn.SiLU(),
                nn.Linear(256, 128),nn.LayerNorm(128),nn.SiLU(),
                nn.Linear(128, 64), nn.LayerNorm(64),nn.SiLU(),
                nn.Linear(64, 2)  # Output 2D score estimate
            )
        
        def forward(self, x, t):
            # Concatenate 2D position and time
            input = torch.cat([x, t], dim=1)
            return self.fc(input)+x

    # Instantiate the score estimator
    denoiser = Denoiser().to(device)

    # Adam optimizer
    optimizer = optim.Adam(denoiser.parameters(), lr=1e-4)
    # Training loop for score matching
    def train_score_estimator(forward_sde, score_estimator, num_epochs=2000, batch_size=1000):
        
        active_batch=batch_size*5
        for epoch in range(num_epochs):
            # Sample 1000 points per character
            # Sample from the camca distribution
            CAMCA=[]
            for i in range (5):
                CAMCA.append(camca_dist.sample(i,batch_size))
            x0 =torch.concat(CAMCA,dim=0).to(device)  

            t = torch.rand(active_batch, 1).to(device)  # Uniformly sample t from [0, 1)
            t=(1.0-1e-2)*t+1e-2
            # Get x_t given x_0 from the forward SDE
            xt = forward_sde.sample_x_t_given_x_0(x0, t)
            
            # Compute the score estimate
            x0_est = score_estimator(xt, t)

            # Score matching loss (MSE between estimated and true score)
            loss = (((x0_est - x0)/t) ** 2).mean()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

        # save the trained score estimator
        torch.save(denoiser.state_dict(), 'denoiser.pth')

    # Train the score estimator
    train_score_estimator(forward_diffusion_SDE, denoiser)

        
        
    # if the weights exist, load them
    try:
        denoiser.load_state_dict(torch.load('denoiser.pth'))
    except:
        print("No weights found for the score estimator. Training the score estimator...")

    
    def score_estimator(x, t):
        x0_pred = denoiser(x, t)
        Sigma = forward_diffusion_SDE.Sigma(t)
        Sigma.scalar += 1e-8  # Add small constant to avoid division by zero
        mu = forward_diffusion_SDE.H(t) @ x0_pred
        return -1.0 * (Sigma.inverse_LinearOperator() @ (x - mu))
    
    with torch.no_grad():
        denoiser.eval()

        # Get the reverse SDE using the trained score estimator
        reverse_diffusion_SDE = forward_diffusion_SDE.reverse_SDE_given_score_estimator(score_estimator)

        # from the end of the Forward Process
        xT=xt[-1].to(device) 

        n_reverse_steps=300
        reverse_timesteps = torch.linspace(1.0, 0.0, n_reverse_steps).to(device)
        xt_reverse = reverse_diffusion_SDE.sample(xT, reverse_timesteps, sampler='euler', return_all=True, verbose=True)

    
    # plot the samples

    fig = plt.figure(figsize=(6, 6))
    ln_C1, = plt.plot(xt_reverse[0][:n, 0], xt_reverse[0][:n, 1], 'o', color='red', markersize=1)
    ln_A1, = plt.plot(xt_reverse[0][n:n*2, 0], xt_reverse[0][n:n*2, 1], 'o', color='blue', markersize=1)
    ln_M, = plt.plot(xt_reverse[0][n*2:n*3, 0], xt_reverse[0][n*2:n*3, 1], 'o', color='green', markersize=1)
    ln_C2, = plt.plot(xt_reverse[0][n*3:n*4, 0], xt_reverse[0][n*3:n*4, 1], 'o', color='purple', markersize=1)
    ln_A2, = plt.plot(xt_reverse[0][n*4:n*5, 0], xt_reverse[0][n*4:n*5, 1], 'o', color='orange', markersize=1)
    plt.xlim(-10, 10)
    plt.ylim(-5, 5)

    # Update function for animation
    def update(frame):

        ln_C1.set_data(xt_reverse[frame][0:n, 0].detach(), xt_reverse[frame][0:n, 1].detach())
        ln_A1.set_data(xt_reverse[frame][n:n*2, 0].detach(), xt_reverse[frame][n:n*2, 1].detach())
        ln_M.set_data(xt_reverse[frame][n*2:n*3, 0].detach(), xt_reverse[frame][n*2:n*3, 1].detach())
        ln_C2.set_data(xt_reverse[frame][n*3:n*4, 0].detach(), xt_reverse[frame][n*3:n*4, 1].detach())
        ln_A2.set_data(xt_reverse[frame][n*4:n*5, 0].detach(), xt_reverse[frame][n*4:n*5, 1].detach())
        
        
        print(f"Animated Frame: {frame}")
        return ln_C1, ln_A1, ln_M,ln_C2, ln_A2


    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(xt_reverse), blit=False)

    # Higher quality video with ffmpeg settings
    writer = animation.writers['ffmpeg'](fps=30, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
    ani.save('camca_reverse_diffusion.mp4', writer=writer, dpi=300)

    print("Done!")
