import torch
import matplotlib.pyplot as plt
from mic_distribution import MIC_Distribution
from diffusion_laboratory.sde import ScalarSDE
# from diffusion_laboratory.linalg import ScalarLinearOperator

# Create the MIC_Distribution instance
mic = MIC_Distribution(bandwidth=0.1)

beta = torch.tensor(3.0)
signal_scale = lambda t: torch.exp(-0.5*beta*t)
noise_variance = lambda t: (1 - torch.exp(-beta*t))
signal_scale_prime = lambda t: -0.5*beta*torch.exp(-0.5*beta*t)
noise_variance_prime = lambda t: beta*torch.exp(-beta*t)

forward_diffusion_SDE = ScalarSDE(  signal_scale=signal_scale,
                                    noise_variance=noise_variance,
                                    signal_scale_prime=signal_scale_prime,
                                    noise_variance_prime=noise_variance_prime)

x0 = mic.sample(1000)
timesteps = torch.linspace(0.0, 1.0, 100)
xt = forward_diffusion_SDE.sample(x0, timesteps, sampler='euler', return_all=True, verbose=True)

# plot the samples
fig = plt.figure(figsize=(6, 6))
ln, = plt.plot(xt[0][:, 0], xt[0][:, 1], 'o', markersize=1)
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# now do an animation

import matplotlib.animation as animation

# Update function for animation
def update(frame):
    ln.set_data(xt[frame][:, 0], xt[frame][:, 1])
    print(f"Animated Frame: {frame}")
    return ln,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(xt), blit=False)

# Higher quality video with ffmpeg settings
writer = animation.writers['ffmpeg'](fps=10, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
ani.save('mic_forward_diffusion.mp4', writer=writer, dpi=300)

print("Done!")