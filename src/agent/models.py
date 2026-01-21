import torch
import torch.nn as nn
import torch.nn.functional as F

class WorldModelCNN(nn.Module):
    """
    Predicts the NEXT STATE based on CURRENT STATE + ACTION.
    Input: Grid (1, 20, 20) + Agent Map (1, 20, 20) + Action Embedding
    Output: Next Grid (1, 20, 20) + Next Energy (Scalar)
    """
    def __init__(self, grid_size=20, num_actions=6):
        super(WorldModelCNN, self).__init__()
        
        self.grid_size = grid_size
        self.num_actions = num_actions
        
        # 1. Action Embedding
        self.action_fc = nn.Linear(num_actions, grid_size * grid_size)
        
        # 2. Encoder (Input is 3 channels: Grid + Agent_Map + Action)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 3. Decoder
        self.up_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.up_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.up_conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
        # 4. Scalar Head (Energy Prediction)
        # NOTE: We use a placeholder here. The first forward pass will detect the real size.
        self._energy_input_size = None
        self.energy_fc = None

    def _init_energy_fc(self, x):
        """Lazy initialization of the energy head to avoid hardcoded size errors."""
        if self.energy_fc is None:
            flatten_size = x.view(x.size(0), -1).size(1)
            self.energy_fc = nn.Sequential(
                nn.Linear(flatten_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ).to(x.device)

    def forward(self, grid_input, agent_map_input, action_input):
        batch_size = grid_input.shape[0]
        
        # 1. Process Action -> Spatial Map
        action_map = self.action_fc(action_input) 
        action_map = action_map.view(batch_size, 1, self.grid_size, self.grid_size)
        
        # 2. Concatenate Inputs (Grid + Agent + Action)
        x = torch.cat([grid_input, agent_map_input, action_map], dim=1)
        
        # 3. Encode
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) 
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) 
        
        x_bottleneck = F.relu(self.conv3(x)) 
        
        # 4. Initialize Energy Head if this is the first run
        if self.energy_fc is None:
            self._init_energy_fc(x_bottleneck)
        
        # 5. Decode Grid
        d = F.interpolate(x_bottleneck, scale_factor=2) 
        d = F.relu(self.up_conv1(d))
        
        d = F.interpolate(d, scale_factor=2) 
        d = F.relu(self.up_conv2(d))
        
        grid_pred = self.up_conv3(d) 
        
        # 6. Predict Energy
        flat = x_bottleneck.view(batch_size, -1)
        energy_pred = self.energy_fc(flat)
        
        return grid_pred, energy_pred
