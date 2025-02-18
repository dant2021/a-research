import torch
import torch.nn as nn
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (Tensor): Window function.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1)
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    real = x_stft.real
    imag = x_stft.imag

    # Calculate magnitude
    mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))
    return mag


class STFTLoss(nn.Module):
    """STFT loss module."""
    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window='hann_window'):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer('window', getattr(torch, window)(win_length))

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Magnitude loss value.
            Tensor: Phase loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        
        # Log magnitude loss
        log_mag_loss = F.l1_loss(torch.log(x_mag), torch.log(y_mag))
        # Magnitude loss
        mag_loss = F.l1_loss(x_mag, y_mag)
        
        return log_mag_loss, mag_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""
    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window='hann_window'):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Ground truth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STF magnitude loss value.
        """
        ret_log_mag = 0
        ret_mag = 0
        for f in self.stft_losses:
            log_mag_loss, mag_loss = f(x, y)
            ret_log_mag += log_mag_loss
            ret_mag += mag_loss
        
        ret_log_mag /= len(self.stft_losses)
        ret_mag /= len(self.stft_losses)

        return ret_log_mag + ret_mag