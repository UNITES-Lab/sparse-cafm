import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# 1. Define the DriftNet
class DriftNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, time_emb_dim):
        super(DriftNet, self).__init__()
        # Time Embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Combined Input: x_t and time embedding
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t):
        """
        x: Tensor of shape (batch_size, input_dim)
        t: Tensor of shape (batch_size, 1)
        """
        t_emb = self.time_emb(t)
        xt = torch.cat([x, t_emb], dim=1)
        return self.net(xt)

# 2. Define the Schrödinger Bridge
class DiffusionSchrodingerBridge:
    def __init__(self, model, device, T=1.0, N=100):
        """
        model: Instance of DriftNet
        device: 'cuda' or 'cpu'
        T: Total time
        N: Number of discretization steps
        """
        self.model = model.to(device)
        self.device = device
        self.T = T
        self.N = N
        self.dt = T / N
    
    def sample_time(self, batch_size):
        """
        Sample a random time t uniformly from [0, T]
        """
        return torch.rand(batch_size, 1, device=self.device) * self.T
    
    def forward_step(self, x, t):
        """
        Perform one forward Euler-Maruyama step
        """
        drift = self.model(x, t)
        noise = torch.randn_like(x)
        x_next = x + drift * self.dt + noise * np.sqrt(self.dt)
        return x_next, drift
    
    def compute_loss(self, x0):
        """
        
        Compute the loss by simulating the forward process and accumulating the drift terms.
        x0: Initial data (batch_size, input_dim)
        """
        batch_size, input_dim = x0.shape
        x = x0.to(self.device)
        loss = 0.0
        
        for _ in range(self.N):
            t = torch.rand(batch_size, 1, device=self.device) * self.T
            drift = self.model(x, t)
            loss += (drift ** 2).sum(dim=1).mean() * self.dt / 2  # L = E[ ∫ ||f_theta||^2 / 2 dt ]
            noise = torch.randn_like(x)
            x = x + drift * self.dt + noise * np.sqrt(self.dt)
        
        return loss
    
    def sample_reverse_process(self, num_samples, device, T=1.0, N=100):
        self.model.eval()
        input_dim = self.model.net[-1].out_features  # Assuming last layer's out_features equals input_dim
        
        with torch.no_grad():
            # Initialize with pure noise
            x = torch.randn(num_samples, input_dim).to(device)
            
            for i in tqdm(range(N)):
                t = torch.rand(num_samples, 1, device=device) * T
                drift = self.model(x, t)
                # Reverse step: x = x - f_theta * dt + noise
                x = x - drift * self.dt + torch.randn_like(x) * np.sqrt(self.dt)
            
            return x

# 3. Visualization Function
def visualize_samples(samples, num_images=16):
    samples = samples.cpu().numpy()
    fig, axes = plt.subplots(1, num_images, figsize=(num_images, 1))
    for i in range(num_images):
        img = samples[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.show()

# 4. Main Execution
def main():
    # Hyperparameters
    batch_size = 128
    image_size = 28
    channels = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = image_size * image_size * channels  # 784 for MNIST
    hidden_dim = 256
    time_emb_dim = 32
    learning_rate = 1e-3
    num_epochs = 10

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten the images
    ])

    # Loading MNIST Dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize DriftNet
    model = DriftNet(input_dim=input_dim, hidden_dim=hidden_dim, time_emb_dim=time_emb_dim)

    # Initialize Schrödinger Bridge
    bridge = DiffusionSchrodingerBridge(model=model, device=device, T=1.0, N=100)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        bridge.model.train()
        epoch_loss = 0.0
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            data = data.to(device)  # Shape: (batch_size, 784)
            
            optimizer.zero_grad()
            loss = bridge.compute_loss(data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # Generate and visualize samples
    num_samples = 16
    generated_samples = bridge.sample_reverse_process(num_samples, device, T=bridge.T, N=bridge.N)
    visualize_samples(generated_samples, num_images=num_samples)

if __name__ == "__main__":
    main()
