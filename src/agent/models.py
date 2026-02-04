import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual 
        return F.relu(out)

class WorldModelResNet(nn.Module):
    def __init__(self, grid_size=20, num_actions=6):
        super(WorldModelResNet, self).__init__()
        
        self.grid_size = grid_size
        self.num_actions = num_actions

        # 1. Action Embedding (Internal)
        # Projects 6-dim action vector -> 20x20 spatial map
        self.action_fc = nn.Linear(num_actions, grid_size * grid_size)

        # 2. Input Embedding (Grid + Agent + ActionMap)
        # Input channels: 1 (Grid) + 1 (Agent) + 1 (ActionMap) = 3
        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        
        # 3. Processing Stack (Resolution stays 20x20)
        self.res_blocks = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )
        
        # 4. Delta Heads
        self.grid_delta_head = nn.Conv2d(32, 1, kernel_size=1) 
        
        # 5. Energy Head
        self.energy_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, grid, agent_map, action_vec):
        # grid: (B, 1, 20, 20)
        # agent_map: (B, 1, 20, 20)
        # action_vec: (B, 6) -> One-hot vector

        # Embed Action to Spatial Map
        batch_size = grid.shape[0]
        action_map = self.action_fc(action_vec)
        action_map = action_map.view(batch_size, 1, self.grid_size, self.grid_size)

        # Stack Inputs
        x = torch.cat([grid, agent_map, action_map], dim=1)
        
        # Process
        x = F.relu(self.input_conv(x))
        x = self.res_blocks(x)
        
        # Predict Deltas
        delta_grid = self.grid_delta_head(x)
        next_energy = self.energy_head(x)
        
        return delta_grid, next_energy
