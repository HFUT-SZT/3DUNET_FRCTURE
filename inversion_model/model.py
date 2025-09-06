import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这一行以使用F.interpolate

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        # Contracting path
        self.encoder1 = self.double_conv(4, 32)
        self.encoder2 = self.double_conv(32, 64)
        self.encoder3 = self.double_conv(64, 128)
        self.encoder4 = self.double_conv(128, 256)
        self.encoder5 = self.double_conv(256, 512)

        # Expanding path
        self.upconv1 = self.up_conv(512, 256)
        self.decoder1 = self.double_conv(512, 256)
        self.upconv2 = self.up_conv(256, 128)
        self.decoder2 = self.double_conv(256, 128)
        self.upconv3 = self.up_conv(128, 64)
        self.decoder3 = self.double_conv(128, 64)
        self.upconv4 = self.up_conv(64, 32)
        self.decoder4 = self.double_conv(64, 32)

        # Final convolution (output 4 channels)
        self.final_conv = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Dropout layer
        self.dropout = nn.Dropout3d(p=0.5)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def up_conv(self, in_channels, out_channels):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(nn.MaxPool3d(2)(e1))
        e3 = self.encoder3(nn.MaxPool3d(2)(e2))
        e4 = self.encoder4(nn.MaxPool3d(2)(e3))
        e5 = self.encoder5(nn.MaxPool3d(2)(e4))

        # Apply dropout after the last encoder block
        e5 = self.dropout(e5)

        # Upconv and decoder steps
        d1 = self.upconv1(e5)
        d1 = self.crop_and_concat(e4, d1)  # Use crop_and_concat for size matching
        d1 = self.decoder1(d1)

        d2 = self.upconv2(d1)
        d2 = self.crop_and_concat(e3, d2)  # Same as above
        d2 = self.decoder2(d2)

        d3 = self.upconv3(d2)
        d3 = self.crop_and_concat(e2, d3)  # Same as above
        d3 = self.decoder3(d3)

        d4 = self.upconv4(d3)
        d4 = self.crop_and_concat(e1, d4)  # Same as above
        d4 = self.decoder4(d4)

        output = self.final_conv(d4)
        return output

    def crop_and_concat(self, encoder_output, decoder_output):
        """ Ensure the encoder output and decoder output have the same size. """
        # Use F.interpolate to match sizes
        if encoder_output.size()[2:] != decoder_output.size()[2:]:
            decoder_output = F.interpolate(decoder_output, size=encoder_output.size()[2:], mode='trilinear', align_corners=True)
        return torch.cat((encoder_output, decoder_output), dim=1)


# Main function to test the model
if __name__ == "__main__":
    # Create the model
    model = UNet3D()

    # Create a dummy input tensor (batch_size, channels, depth, height, width)
    input_tensor = torch.randn(1, 4, 96, 72, 24)

    # Forward pass
    output = model(input_tensor)

    # Print input and output shapes
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
