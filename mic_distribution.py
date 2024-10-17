import torch

class MIC_Distribution:
    def __init__(self, bandwidth=0.1):
        self.bandwidth = bandwidth  # Standard deviation for isotropic Gaussians
        self.points_M = self.create_M()
        self.points_I = self.create_I()
        self.points_C = self.create_C()
        self.points = torch.cat([self.points_M, self.points_I, self.points_C], dim=0)  # All points for MIC
        self.points -= torch.mean(self.points, dim=0, keepdim=True)
        self.points *= 3.0 # Scale the points to have a larger spread
        self.num_points = self.points.size(0)
    
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
    
    def sample(self, n_samples=100):
        """Samples n points from the Gaussian mixture model."""
        # Sample random indices for the points
        indices = torch.randint(0, self.num_points, (n_samples,))
        chosen_points = self.points[indices]

        # Sample isotropic Gaussians around the chosen points
        samples = chosen_points + torch.randn(n_samples, 2) * self.bandwidth
        return samples

    def log_pdf(self, x):
        """Computes the log probability density function of the Gaussian mixture at points x."""
        x = x.unsqueeze(1)  # [n_samples, 1, 2]
        diffs = x - self.points.unsqueeze(0)  # [n_samples, num_points, 2]
        exponent = -0.5 * torch.sum(diffs ** 2, dim=2) / (self.bandwidth ** 2)
        log_probs = exponent - torch.log(2 * torch.pi * self.bandwidth ** 2)
        return torch.logsumexp(log_probs, dim=1) - torch.log(torch.tensor(self.num_points))

    def score(self, x):
        """Computes the score, which is the gradient of the log probability density function."""
        x = x.unsqueeze(1)  # [n_samples, 1, 2]
        diffs = x - self.points.unsqueeze(0)  # [n_samples, num_points, 2]
        exponent = -0.5 * torch.sum(diffs ** 2, dim=2) / (self.bandwidth ** 2)
        log_probs = exponent - torch.log(torch.tensor(2 * torch.pi * self.bandwidth ** 2))
        weights = torch.softmax(log_probs, dim=1)  # Softmax to get weights
        score = torch.sum(weights.unsqueeze(2) * (-diffs / (self.bandwidth ** 2)), dim=1)
        return score


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create the MIC_Distribution instance
    mic = MIC_Distribution(bandwidth=0.1)

    # Sample 1000 points
    samples = mic.sample(10000)

    # Plot the sampled points
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=2, alpha=0.6)
    plt.title("2D Gaussian Mixture Model Sampling for 'MIC'")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("mic_distribution.png")
