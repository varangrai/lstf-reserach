import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import functional as f


class EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = f.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = f.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + input_

        return attention

class Model(nn.Module):
    def __init__(self, configs, hidden_dim=128):
        super(Model, self).__init__()
        # Dimensions
        self.L = configs.seq_len  # seq len 
        self.S = configs.pred_len  # output len
        self.num_channels = 7
        
        # CI linear layers for each channel
        self.ci_linears = nn.ModuleList([nn.Linear(self.L, self.S) for _ in range(self.num_channels)])
        
        # CD linear layer
        self.cd_linear = nn.Linear(self.L, self.S)
        
        # Efficient Attention mechanism for CI and CD modulation
        # The input channels are doubled due to concatenation of CI and CD
        self.efficient_attention = EfficientAttention(2 * self.S, self.num_channels, 1, self.S)
            
        # Final prediction layer
        self.prediction = nn.Linear(2*self.S, self.S)
        
    def forward(self, x):
      x = x.permute(0, 2, 1)  # Shape: (batch_size, num_channels, L)
      
      # CI transformations
      ci_outputs = [self.ci_linears[i](x[:, i, :]) for i in range(self.num_channels)]
      x_ci = torch.stack(ci_outputs, dim=1)  # Shape: (batch_size, num_channels, S)
      
      # CD transformation
      x_cd = self.cd_linear(x)  # Shape: (batch_size, num_channels, S)
      
      # Combine CI and CD
      combined = torch.cat((x_ci, x_cd), dim=-1)  # Shape: (batch_size, num_channels, 2*S)
      
      # Reshape for EfficientAttention
      combined = combined.permute(0, 2, 1).unsqueeze(3)  # Shape: (batch_size, 2*S, num_channels, 1)
      
      # Efficient Attention modulation
      context = self.efficient_attention(combined)
      
      # Remove the dummy spatial dimension and reshape
      context = context.squeeze(3).permute(0, 2, 1)  # Shape: (batch_size, num_channels, 2*S)    
      
      # Final prediction
      out = self.prediction(context).permute(0,2,1)  # Shape: (batch_size, S, num_channels)
      
      return out




# Loss function combining MAE and Huber loss
def combined_loss(y_pred, y_true, sigma=1.0):
    huber = F.smooth_l1_loss(y_pred, y_true, reduction='none')
    mae = torch.abs(y_pred - y_true)
    loss = torch.where(torch.abs(y_pred - y_true) < sigma, huber, mae)
    return loss.mean()

# Example usage
# num_channels = 10
# model = MLinear(num_channels=num_channels, L=100, S=10, hidden_dim=20)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Dummy data
# x = torch.randn(32, num_channels, 100)  # L=100
# y_true = torch.randn(32, 10)  # S=10

# # Forward pass
# y_pred = model(x)

# # Compute loss
# loss = combined_loss(y_pred, y_true)

# # Backward pass and optimization
# loss.backward()
# optimizer.step()
