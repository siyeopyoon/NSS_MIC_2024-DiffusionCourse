import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mic_distribution import MIC_Distribution
from diffusion_laboratory.sde import ScalarSDE
import matplotlib.animation as animation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# Create the MIC_Distribution instance
mic = MIC_Distribution(bandwidth=0.1)

beta = torch.tensor(3.0)
signal_scale = lambda t: torch.exp(-0.5 * beta * t)
noise_variance = lambda t: (1 - torch.exp(-beta * t))
signal_scale_prime = lambda t: -0.5 * beta * torch.exp(-0.5 * beta * t)
noise_variance_prime = lambda t: beta * torch.exp(-beta * t)

# Define the forward diffusion process
forward_diffusion_SDE = ScalarSDE(
    signal_scale=signal_scale,
    noise_variance=noise_variance,
    signal_scale_prime=signal_scale_prime,
    noise_variance_prime=noise_variance_prime
)

# Sample initial points
x0 = mic.sample(1000)
timesteps = torch.linspace(0.0, 1.0, 100)

# Define the score estimator network (3 -> 2 dense NN)
class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
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
    for epoch in range(num_epochs):
        x0 = mic.sample(batch_size).to(device)  # Sample from the MIC distribution
        t = T*torch.rand(batch_size, 1).to(device)  # Uniformly sample t from [0, 1)
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
        
        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")


# if the weights exist, load them
try:
    denoiser.load_state_dict(torch.load('denoiser.pth'))
except:
    print("No weights found for the score estimator. Training the score estimator...")

# Train the score estimator
train_score_estimator(forward_diffusion_SDE, denoiser)

# save the trained score estimator
torch.save(denoiser.state_dict(), 'denoiser.pth')


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
xT = torch.randn(1000, 2).to(device)

reverse_timesteps = torch.linspace(T, 0.0, 100).to(device)
denoiser.eval()
xt_reverse = reverse_diffusion_SDE.sample(xT, reverse_timesteps, sampler='euler', return_all=True, verbose=True)

# Plot the reverse diffusion samples
fig = plt.figure(figsize=(6, 6))
ln, = plt.plot(xt_reverse[0][:, 0], xt_reverse[0][:, 1], 'o', markersize=1)
plt.xlim(-5, 5)
plt.ylim(-5, 5)

# Update function for animation
def update(frame):
    ln.set_data(xt_reverse[frame][:, 0].detach(), xt_reverse[frame][:, 1].detach())
    print(f"Animated Frame: {frame}")
    return ln,

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(xt_reverse), blit=False)

# Higher quality video with ffmpeg settings
writer = animation.writers['ffmpeg'](fps=10, bitrate=5000, extra_args=['-pix_fmt', 'yuv420p', '-crf', '15'])
ani.save('mic_reverse_diffusion.mp4', writer=writer, dpi=300)

print("Done!")
