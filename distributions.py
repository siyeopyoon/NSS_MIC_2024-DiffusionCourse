import torch


class Spiral_Distribution:
    def __init__(self, bandwidth=0.1):
        self.bandwidth = bandwidth  # Standard deviation for isotropic Gaussians
        theta_max = 4 * torch.pi  # Max angle in radians (2 turns)

        radius = 0.5  # Radius of the spiral

        # Generate points using PyTorch
        theta = torch.linspace(0, theta_max, 50)  # Angular position
        z = torch.linspace(-5, 5, 50)  # Height along the z-axis
        x = radius*abs(z+5) * torch.cos(theta)  # X-coordinates of the spiral
        y = radius*abs(z+5) * torch.sin(theta)  # Y-coordinates of the spiral

        
        # Stack x, y, and z to form a 3D tensor (n_points, 3)
        self.points_all=torch.stack((x, y, z), dim=1)
        
        self.num_points =self.points_all.size(0)
 
    
    def sample(self, n_samples=100):
        """Samples n points from the Gaussian mixture model."""
        # Sample random indices for the points
        indices = torch.randint(0, self.points_all.size(0), (n_samples,))
        chosen_points = self.points_all[indices]

        # Sample isotropic Gaussians around the chosen points
        samples = chosen_points + torch.randn(n_samples, 3) * self.bandwidth
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
        
        
        self.num_points =self.points_all.size(0)
    
    
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


class Saddle_Distribution:
    def __init__(self, bandwidth=0.1):
        self.bandwidth = bandwidth  # Standard deviation for isotropic Gaussians
        # Generate a grid of x, y values
        n_points = 50
        x = torch.linspace(-3, 3, n_points)
        y = torch.linspace(-3, 3, n_points)
        x, y = torch.meshgrid(x, y)

        # Saddle surface equation: z = x^2 - y^2
        z = 1/4*(x**2 - y**2)
        # Flatten x, y, z tensors to 1D
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()


        # Stack x, y, and z to form a 3D tensor (n_points, 3)
        self.points_all=torch.stack([x_flat, y_flat, z_flat], dim=1)
        
        self.num_points =self.points_all.size(0)
 
    
    def sample(self, n_samples=100):
        """Samples n points from the Gaussian mixture model."""
        # Sample random indices for the points
        indices = torch.randint(0, self.points_all.size(0), (n_samples,))
        chosen_points = self.points_all[indices]

        # Sample isotropic Gaussians around the chosen points
        samples = chosen_points + torch.randn(n_samples, 3) * self.bandwidth
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
    
    

