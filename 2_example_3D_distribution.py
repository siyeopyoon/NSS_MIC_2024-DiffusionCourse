import torch
import torch.nn as nn
import torch.optim as optim
from diffusion_laboratory.sde import StochasticDifferentialEquation
from diffusion_laboratory.linalg import ScalarLinearOperator

from diffusion_laboratory.sde import ScalarSDE
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation
import distributions


step_size = 0.5



    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create the 3D point distribution instance
    
    keyword="spiral"
    if keyword == "saddle":
        dist = distributions.Saddle_Distribution(bandwidth=0.1)
    elif keyword== "spiral":
        dist = distributions.Spiral_Distribution(bandwidth=0.1)
    
    n=1000
    # Sample 1000 points
    samples= dist.sample(n)
   
   # plot the samples
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Initial scatter plot (3D points)
    scatter = ax.scatter(samples[:n, 0], samples[:n, 1], samples[:n, 2], label=f"3D {keyword} (PyTorch)")

    # Add labels
    plt.legend(loc="upper right")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    

    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f"{keyword}_distribution.png")

        
    def f(x, t): 
        return 0.5*step_size*step_size*dist.score(x)

    def G(x, t): 
        return ScalarLinearOperator(step_size)

        
    langevin_dynamics_SDE = StochasticDifferentialEquation(f=f, G=G)

    timesteps = torch.linspace(0.0, 1.0, 100)
    xt = langevin_dynamics_SDE.sample(samples, timesteps, sampler='euler', return_all=True, verbose=True)

    print (len(xt))
    
    # plot the samples
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Initial scatter plot (3D points)
    scatter = ax.scatter(xt[0][:n, 0], xt[0][:n, 1], xt[0][:n, 2], label=f"3D {keyword} (PyTorch)")

    # Add labels
    plt.legend(loc="upper right")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    # Update function for animation (for scatter points)
    def update(frame):
        ax.clear()  # Clear the previous frame

        # Update scatter plot with new data
        scatter = ax.scatter(xt[frame][:n, 0], xt[frame][:n, 1], xt[frame][:n, 2], label=f"3D {keyword} (PyTorch)")
        
        # Add axis labels and legend back
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_zlim(-5,5)
        plt.legend(loc="upper right")
        
        print(f"Animated Frame: {frame}")
        return scatter,
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(timesteps), blit=False)

    # Save the animation using ffmpeg
    writer = animation.FFMpegWriter(fps=10, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
    ani.save(f'{keyword}_langevin_dynamics.mp4', writer=writer, dpi=150)

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
    ax = fig.add_subplot(111, projection='3d')

    # Initial scatter plot (3D points)
    scatter = ax.scatter(xt[0][:n, 0], xt[0][:n, 1], xt[0][:n, 2], label=f"3D {keyword} (PyTorch)")

    # Add labels
    plt.legend(loc="upper right")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(0,10)
    # Update function for animation (for scatter points)
    def update(frame):
        ax.clear()  # Clear the previous frame

        # Update scatter plot with new data
        scatter = ax.scatter(xt[frame][:n, 0], xt[frame][:n, 1], xt[frame][:n, 2], label=f"3D {keyword} (PyTorch)")
        
        # Add axis labels and legend back
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_zlim(-5,5)
        plt.legend(loc="upper right")
        
        print(f"Animated Frame: {frame}")
        return scatter,
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(timesteps), blit=False)

    # Save the animation using ffmpeg
    writer = animation.FFMpegWriter(fps=10, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
    ani.save(f'{keyword}_forward_dynamics.mp4', writer=writer, dpi=300)

    print("Done!")
    
    
        
        
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timesteps = torch.linspace(0.0, 1.0, 100)

    # Define the score estimator network (3 -> 2 dense NN)
    class Denoiser(nn.Module):
        def __init__(self):
            super(Denoiser, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(4, 64), nn.LayerNorm(64),nn.SiLU(),
                nn.Linear(64, 128),nn.LayerNorm(128),nn.SiLU(),
                nn.Linear(128, 64),nn.LayerNorm(64),nn.SiLU(),
                nn.Linear(64, 3)  # Output 2D score estimate
            )
        
        def forward(self, x, t):
            # Concatenate 2D position and time
            input = torch.cat([x, t], dim=1)
            out=x+self.fc(input)
            return out
        

    # Instantiate the score estimator
    denoiser = Denoiser().to(device)

    # Adam optimizer
    optimizer = optim.Adam(denoiser.parameters(), lr=1e-3)

    T = 0.5
    # Training loop for score matching
    def train_score_estimator(forward_sde, score_estimator, num_epochs=3000, batch_size=100):
        
        for epoch in range(num_epochs):
            # Sample 1000 points
            x0 =dist.sample(batch_size).to(device)  # Sample from the distribution
            t = T*torch.rand(batch_size, 1).to(device)  # Uniformly sample t from [0, 1)
            t = t**2.0 # square it to emphasize early time points

            # Get x_t given x_0 from the forward SDE
            xt = forward_sde.sample_x_t_given_x_0(x0, t)
            
            # Compute the score estimate
            x0_est = score_estimator(xt, t)

            # Score matching loss (MSE between estimated and true score)
            loss = ((x0_est - x0) ** 2).mean()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")

        # save the trained score estimator
        torch.save(denoiser.state_dict(), f'denoiser_{keyword}.pth')
        
    # Train the score estimator
    train_score_estimator(forward_diffusion_SDE, denoiser)
    
    # if the weights exist, load them
    try:
        denoiser.load_state_dict(torch.load(f'denoiser_{keyword}.pth'))
    except:
        print("No weights found for the score estimator. Training the score estimator...")

    




    def score_estimator(x, t):
        x0_pred = denoiser(x, t)
        Sigma = forward_diffusion_SDE.Sigma(t)
        Sigma.scalar += 1e-8  # Add small constant to avoid division by zero
        mu = forward_diffusion_SDE.H(t) @ x0_pred
        return -1.0 * (Sigma.inverse_LinearOperator() @ (x - mu))


    # Get the reverse SDE using the trained score estimator
    reverse_diffusion_SDE = forward_diffusion_SDE.reverse_SDE_given_score_estimator(score_estimator)
    
    # Start from the last step of forward 
    xT=xt[-1].to(device)
    
    # or purely new Gaussian
    # xT = torch.randn(1000, 3).to(device)
    
    # or repeat the forward again
    # x0 = dist.sample(1000)
    # xT = forward_diffusion_SDE.sample_x_t_given_x_0(x0, torch.tensor(1.0).to(device))
    
    # Sample points using the reverse SDE
    reverse_steps=100
    reverse_timesteps = torch.linspace(T, 0.0, reverse_steps).to(device)
    denoiser.eval()
    xt_reverse = reverse_diffusion_SDE.sample(xT, reverse_timesteps, sampler='euler', return_all=True, verbose=True)

 
    # plot the samples
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Initial scatter plot (3D points)
    scatter = ax.scatter(xt_reverse[0][:n, 0], xt_reverse[0][:n, 1], xt_reverse[0][:n, 2], label="3D Spiral (PyTorch)")

    # Add labels
    plt.legend(loc="upper right")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    # Update function for animation (for scatter points)
    def update(frame):
        ax.clear()  # Clear the previous frame

        # Update scatter plot with new data
        scatter = ax.scatter(xt_reverse[frame][:n, 0].detach(), 
                             xt_reverse[frame][:n, 1].detach(), 
                             xt_reverse[frame][:n, 2].detach(), label=f"3D {keyword} (PyTorch)")
        
        # Add axis labels and legend back
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        ax.set_zlim(-5,5)
        plt.legend(loc="upper right")
        
        print(f"Animated Frame: {frame}")
        return scatter,
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(timesteps), blit=False)

    # Save the animation using ffmpeg
    writer = animation.FFMpegWriter(fps=10, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
    ani.save(f'{keyword}_reverse_dynamics.mp4', writer=writer, dpi=300)

    print("Done!")
    
    
        
