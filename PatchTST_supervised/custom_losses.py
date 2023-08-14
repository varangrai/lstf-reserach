import torch
import torch.nn as nn
import torch.fft

class FFTMSELoss(nn.Module):
    def __init__(self):
        super(FFTMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Compute the FFT of both tensors
        y_pred_fft = torch.fft.fft(y_pred, dim=-1)
        y_true_fft = torch.fft.fft(y_true, dim=-1)
        
        # Compute the MSE between the absolute values of the FFT representations
        loss = self.mse_loss(y_pred_fft.abs(), y_true_fft.abs())
        return loss


