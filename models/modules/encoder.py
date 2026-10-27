import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

from .residual import ResidualBlock
from .attention import AttentionBlock


class VAE_Encoder(nn.Module):
    """
    Encoder that exposes the last spatial feature map before μ/σ projection.

    forward(x, noise, return_features=False):
        returns (z, mean, log_variance) if return_features == False
        returns (z, mean, log_variance, feats) if return_features == True

    Shapes for 256x256 input:
        feats: (B, 512, 32, 32)
        mean, log_variance, z: (B, 4, 32, 32)   # 0.18215 applied to z only
    """

    def __init__(self):
        super().__init__()

        # Stage 1
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, padding=1)                # (B, 3, H, W) -> (B, 128, H, W)
        self.res1_1 = ResidualBlock(128, 128)
        self.res1_2 = ResidualBlock(128, 128)
        self.down1  = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)     # -> (B, 128, H/2, W/2)

        # Stage 2
        self.res2_1 = ResidualBlock(128, 256)                                     # -> (B, 256, H/2, W/2)
        self.res2_2 = ResidualBlock(256, 256)
        self.down2  = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0)     # -> (B, 256, H/4, W/4)

        # Stage 3
        self.res3_1 = ResidualBlock(256, 512)                                     # -> (B, 512, H/4, W/4)
        self.res3_2 = ResidualBlock(512, 512)
        self.down3  = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0)     # -> (B, 512, H/8, W/8)

        # Bottleneck (H/8, W/8)
        self.res4_1 = ResidualBlock(512, 512)
        self.res4_2 = ResidualBlock(512, 512)
        self.res4_3 = ResidualBlock(512, 512)
        self.attn    = AttentionBlock(512)
        self.res4_4 = ResidualBlock(512, 512)
        self.norm    = nn.GroupNorm(32, 512)
        self.act     = nn.SiLU()

        # Projection to params: keep μ/logσ² at 4 channels each
        self.to_param_pre = nn.Conv2d(512, 8, kernel_size=3, padding=1)           # -> (B, 8, H/8, W/8)
        self.to_param     = nn.Conv2d(8,   8, kernel_size=1, padding=0)           # -> (B, 8, H/8, W/8)

    @staticmethod
    def _maybe_downsample_pad(x: torch.Tensor, layer: nn.Module) -> torch.Tensor:
        # Asymmetric padding before stride-2 conv to match original alignment
        if getattr(layer, 'stride', None) == (2, 2):
            # Pad (left, right, top, bottom)
            x = F.pad(x, (0, 1, 0, 1))
        return x

    def forward(
        self,
        x: torch.Tensor,                 # (B,3,H,W)
        noise: torch.Tensor,             # (B,4,H/8,W/8), N(0,1) or zeros if deterministic
        return_features: bool = False
    ):
        # Stage 1
        x = self.conv_in(x)
        x = self.res1_1(x)
        x = self.res1_2(x)
        x = self._maybe_downsample_pad(x, self.down1)
        x = self.down1(x)

        # Stage 2
        x = self.res2_1(x)
        x = self.res2_2(x)
        x = self._maybe_downsample_pad(x, self.down2)
        x = self.down2(x)

        # Stage 3
        x = self.res3_1(x)
        x = self.res3_2(x)
        x = self._maybe_downsample_pad(x, self.down3)
        x = self.down3(x)

        # Bottleneck
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.res4_3(x)
        x = self.attn(x)
        x = self.res4_4(x)
        x = self.norm(x)
        x = self.act(x)

        feats = x  # (B, 512, H/8, W/8) — rich map for template matching

        # Project to μ / logσ²
        x = self.to_param_pre(x)   # (B, 8, H/8, W/8)
        x = self.to_param(x)       # (B, 8, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)  # (B,4,H/8,W/8) each

        # Clamp & reparam
        log_variance = torch.clamp(log_variance, -30.0, 20.0)
        stdev = (log_variance.exp()).sqrt()
        z = mean + stdev * noise
        z = z * 0.18215  # stable-diffusion scaling (keep behavior identical)

        if return_features:
            return z, mean, log_variance, feats
        return z, mean, log_variance


class VQVAE_Encoder(nn.Sequential):

    def __init__(self, latent_dim=128):
        super().__init__(
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height, Width)
            ResidualBlock(128, 128),
            
            # (Batch_Size, 128, Height, Width) -> (Batch_Size, 128, Height / 2, Width / 2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # (Batch_Size, 128, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            ResidualBlock(128, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 2, Width / 2)
            ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height / 4, Width / 4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 256, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            ResidualBlock(256, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            AttentionBlock(512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.GroupNorm(32, 512), 
            
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 8, Width / 8)
            nn.SiLU(), 

            # Because the padding=1, it means the width and height will increase by 2
            # Out_Height = In_Height + Padding_Top + Padding_Bottom
            # Out_Width = In_Width + Padding_Left + Padding_Right
            # Since padding = 1 means Padding_Top = Padding_Bottom = Padding_Left = Padding_Right = 1,
            # Since the Out_Width = In_Width + 2 (same for Out_Height), it will compensate for the Kernel size of 3
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8). 
            nn.Conv2d(512, 8, kernel_size=3, padding=1), 

            # (Batch_Size, 8, Height / 8, Width / 8) -> (Batch_Size, 8, Height / 8, Width / 8)
            nn.Conv2d(8, latent_dim, kernel_size=1, padding=0), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, Out_Channels, Height / 8, Width / 8)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)

        for module in self:

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        # (Batch_Size, 8, Height / 8, Width / 8) 
        return x