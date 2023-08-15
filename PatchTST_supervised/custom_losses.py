import torch
import torch.nn as nn
import torch.fft

class FFTAbsMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Compute the FFT of both tensors
        y_pred_fft = torch.fft.fft(y_pred, dim=-1)
        y_true_fft = torch.fft.fft(y_true, dim=-1)
        
        # Compute the MSE between the absolute values of the FFT representations
        loss = self.mse_loss(y_pred_fft.abs(), y_true_fft.abs())
        return loss

class FFTRealMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Compute the FFT of both tensors
        y_pred_fft = torch.fft.fft(y_pred, dim=-1).real
        y_true_fft = torch.fft.fft(y_true, dim=-1).real
        
        # Compute the MSE between the absolute values of the FFT representations
        loss = self.mse_loss(y_pred_fft, y_true_fft)
        return loss

class CombinedMSELoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha  # Weight for standard MSE
        self.beta = 1 - alpha  # Weight for FFT-based MSE

    def forward(self, y_pred, y_true):
        # Standard MSE
        time_domain_loss = self.mse_loss(y_pred, y_true)

        # FFT-based MSE
        y_pred_fft = torch.fft.fft(y_pred, dim=-1)
        y_true_fft = torch.fft.fft(y_true, dim=-1)
        frequency_domain_loss = self.mse_loss(y_pred_fft.abs(), y_true_fft.abs())

        # Combined loss
        combined_loss = self.alpha * time_domain_loss + self.beta * frequency_domain_loss
        return combined_loss