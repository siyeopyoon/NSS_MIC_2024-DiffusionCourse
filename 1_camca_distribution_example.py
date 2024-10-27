import torch
import torch.nn as nn
import torch.optim as optim
from diffusion_laboratory.sde import StochasticDifferentialEquation
from diffusion_laboratory.linalg import ScalarLinearOperator

from diffusion_laboratory.sde import ScalarSDE


import matplotlib.animation as animation
import distributions

    
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create the camca_Distribution instance
    camca_dist = distributions.CAMCA_Distribution(bandwidth=0.1)
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
    
    
    