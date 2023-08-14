import torch
import torch.nn as nn
import torch.fft

class FFTMSELoss(nn.Module):
    def __init__(self):
        super(FFTMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        # Compute the FFT of both tensors
        y_pred_fft = torch.fft.fft(y_pred, dim=-1).real  # Taking only the real part
        y_true_fft = torch.fft.fft(y_true, dim=-1).real  # Taking only the real part
        
        # Compute the MSE between the FFT representations
        loss = self.mse_loss(y_pred_fft, y_true_fft)
        return loss

