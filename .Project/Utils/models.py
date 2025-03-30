"""
Author: Graham Pellegrini
Date: 2025-03-13
Project: UOM Final Year Project (FYP)

This util module is used within both piplines for training and denoising. It holds the various models trained and tested for the project. The model architectures are defined as classes that inherit from PyTorch's nn.Module class. The models are defined in a separate module to keep the main pipeline scripts clean and focused on the training and evaluation process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# === Simple CNN Architecture ===
# This is a simple CNN architecture using 3 convolutional layers and 2 fully connected layers. The foward pass moves through an encoder and decoder structure, with a bottleneck layer in between for feature extraction. The Architecure is used for the baseline model in the project and focus on the correct in and output shapes for the spectrogram data.
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Encoder (Downsampling)
        self.encoder = nn.Sequential(
            # Input: 2 channels (real and imaginary)
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Bottleneck Layer
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Decoder (Upsampling)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # Output (real+imag)
            nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1),
            # Normalize output between -1 and 1
            nn.Tanh()  
        )

    def forward(self, real, imag):
        # Store original size for upsampling
        orig_size = real.shape[2:]

        # Stack real and imaginary as two-channel input
        x = torch.cat((real, imag), dim=1)

        # Pass through CNN
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)

        # Ensure output size matches the original input
        x = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)

        # Split back into real and imaginary parts
        out_real, out_imag = torch.chunk(x, 2, dim=1)
        return out_real, out_imag

# === CNN model (CED Style) ===
class CNN_CED(nn.Module):
    def __init__(self):
        super(CNN_CED, self).__init__()

        # Encoder (Conv → BN → ReLU → MaxPool) × 5
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 12, kernel_size=(13, 1), padding=(6, 0)),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(12, 16, kernel_size=(11, 1), padding=(5, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(16, 20, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(20, 24, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(24, 32, kernel_size=(5, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Decoder (Conv → BN → ReLU → Upsample) × 5
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=(2, 1)),
            nn.Conv2d(32, 24, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(24),
            nn.ReLU(),

            nn.Upsample(scale_factor=(2, 1)),
            nn.Conv2d(24, 20, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(20),
            nn.ReLU(),

            nn.Upsample(scale_factor=(2, 1)),
            nn.Conv2d(20, 16, kernel_size=(11, 1), padding=(5, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Upsample(scale_factor=(2, 1)),
            nn.Conv2d(16, 12, kernel_size=(13, 1), padding=(6, 0)),
            nn.BatchNorm2d(12),
            nn.ReLU()
        )

        # Final Conv Layer to get 2-channel output (real + imag)
        self.output_layer = nn.Conv2d(12, 2, kernel_size=(8, 1), padding=(4, 0))
        self.activation = nn.Tanh()

    def forward(self, real, imag):
        orig_size = real.shape[2:]
        x = torch.cat((real, imag), dim=1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_layer(x)
        x = self.activation(x)
        x = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)
        out_real, out_imag = torch.chunk(x, 2, dim=1)
        return out_real, out_imag
    

# === R-CED model (Fully Conv, No Pooling/Upsampling) ===
class RCED(nn.Module):
    def __init__(self):
        super(RCED, self).__init__()

        filters = [12, 16, 20, 24, 32, 24, 20, 16, 12]
        kernels = [13, 11, 9, 7, 7, 9, 11, 13, 8]
        layers = []

        in_channels = 2  # Input real + imag
        for out_channels, k in zip(filters, kernels):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=(k, 1), padding=(k // 2, 0)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

        # Final output layer (2 channels: real + imag)
        layers.append(nn.Conv2d(in_channels, 2, kernel_size=(129, 1), padding=(129 // 2, 0)))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self, real, imag):
        orig_size = real.shape[2:]
        x = torch.cat((real, imag), dim=1)
        x = self.network(x)
        x = F.interpolate(x, size=orig_size, mode="bilinear", align_corners=False)
        out_real, out_imag = torch.chunk(x, 2, dim=1)
        return out_real, out_imag


# === UNet Architecture ===
# This is a U-Net architecture usally used for image segmentation. The model is adapted for the spectrogram data by changing the input and output channels to 2 (real and imaginary). The model uses GroupNorm instead of BatchNorm for memory efficiency and PReLU activations. The model uses skip connections to improve the flow of information between the encoder and decoder. The model promises better performance than the simple CNN architecture, due to the skip connections and deeper architecture.
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(2, 64)
        self.enc2 = self.conv_block(64, 128, stride=2)
        self.enc3 = self.conv_block(128, 256, stride=2)
        self.enc4 = self.conv_block(256, 512, stride=2)
        self.enc5 = self.conv_block(512, 1024, stride=2)  # ✨ new

        # Bottleneck
        self.bottleneck = self.conv_block(1024, 1024)

        # Decoder
        self.dec5 = self.deconv_block(1024 + 1024, 512)
        self.dec4 = self.deconv_block(512 + 512, 256)
        self.dec3 = self.deconv_block(256 + 256, 128)
        self.dec2 = self.deconv_block(128 + 128, 64)
        self.dec1 = self.deconv_block(64 + 64, 32)

        # Output Layer (no Tanh for now)
        self.out_layer = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Identity()  # 👈 Replace with Tanh() after tuning if needed

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),  # 👈 Updated normalization
            nn.PReLU()
        )

    def deconv_block(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, real, imag):
        orig_size = real.shape[2:]
        x = torch.cat((real, imag), dim=1)

        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        b = self.bottleneck(e5)

        # Decoder with resizing
        d5 = self.dec5(torch.cat((b, e5), dim=1))
        d4 = self.dec4(torch.cat((d5, self._resize(e4, d5)), dim=1))
        d3 = self.dec3(torch.cat((d4, self._resize(e3, d4)), dim=1))
        d2 = self.dec2(torch.cat((d3, self._resize(e2, d3)), dim=1))
        d1 = self.dec1(torch.cat((d2, self._resize(e1, d2)), dim=1))

        out = self.activation(self.out_layer(d1))
        out = F.interpolate(out, size=orig_size, mode="bilinear", align_corners=False)
        out_real, out_imag = torch.chunk(out, 2, dim=1)
        return out_real, out_imag

    def _resize(self, x, target):
        return F.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)

# === Conv-TasNet Architecture ===
class ConvTasNet(nn.Module):
    def __init__(self, enc_dim=128, feature_dim=48, kernel_size=(3, 3), num_layers=4, num_stacks=2):
        super(ConvTasNet, self).__init__()

        # ✅ Ensure dynamic padding for kernel size consistency
        pad = (kernel_size[0] // 2, kernel_size[1] // 2)

        # Encoder (takes 2 channels: real + imaginary STFT)
        self.encoder = nn.Conv2d(2, enc_dim, kernel_size=kernel_size, padding=pad, bias=False)

        # Separation Network (TCN-based)
        self.separation_net = TemporalConvNet(enc_dim, feature_dim, num_layers, num_stacks)

        # Decoder (outputs 2 channels: denoised real + imaginary STFT)
        self.decoder = nn.Conv2d(enc_dim, 2, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, noisy_real, noisy_imag):
        # Combine real and imaginary parts as two-channel input
        noisy_complex = torch.cat([noisy_real, noisy_imag], dim=1)  # Shape: (batch, 2, freq, time)

        # Pass through encoder
        encoded = self.encoder(noisy_complex)

        # Pass through separation network
        separated = self.separation_net(encoded)

        # Decode back to real + imaginary components
        denoised_real, denoised_imag = torch.chunk(self.decoder(separated), 2, dim=1)

        # ✅ Ensure consistent shapes by cropping if necessary
        min_T = min(denoised_real.shape[-1], noisy_real.shape[-1])
        min_F = min(denoised_real.shape[-2], noisy_real.shape[-2])

        denoised_real = denoised_real[:, :, :min_F, :min_T]
        denoised_imag = denoised_imag[:, :, :min_F, :min_T]

        return denoised_real, denoised_imag

# === Temporal Convolutional Network (TCN) ===
class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, feature_dim, num_layers, num_stacks):
        super(TemporalConvNet, self).__init__()

        # Create TCN with residual blocks
        layers = []
        for stack in range(num_stacks):
            for layer in range(num_layers):
                dilation = 2 ** layer  # Increasing dilation to capture long-range dependencies
                layers.append(ResidualBlock(input_dim, feature_dim, dilation))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# === Residual Block for Temporal Convolutions ===
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, feature_dim, dilation):
        super(ResidualBlock, self).__init__()

        pad = dilation  # Ensure padding accounts for dilation
        self.conv1 = nn.Conv2d(input_dim, feature_dim, kernel_size=(3, 3), padding=pad, dilation=dilation)
        self.conv2 = nn.Conv2d(feature_dim, input_dim, kernel_size=(3, 3), padding=pad, dilation=dilation)

        self.prelu = nn.PReLU()
        self.norm = nn.GroupNorm(8, feature_dim)

    def forward(self, x):
        out = self.prelu(self.conv1(x))
        out = self.norm(out)
        out = self.conv2(out)

        # ✅ Ensure residual connection has matching shape
        min_F = min(x.shape[-2], out.shape[-2])
        min_T = min(x.shape[-1], out.shape[-1])

        out = out[:, :, :min_F, :min_T]
        x = x[:, :, :min_F, :min_T]

        return x + out  # Skip connection
