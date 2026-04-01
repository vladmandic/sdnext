# original: <https://github.com/somanchiu/ReSwapper/blob/main/StyleTransferModel_128.py>

import torch
import torch.nn as nn
import torch.nn.functional as F

class ReSwapperModel(nn.Module):
    def __init__(self):
        super().__init__()

        # self.pad = nn.ReflectionPad2d(3)
        # Encoder for target face
        self.target_encoder = nn.Sequential(
            # self.pad,
            nn.Conv2d(3, 128, kernel_size=7, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        # for style_block in self.target_encoder:
        #     for param in style_block.parameters():
        #         param.requires_grad = False

        # Style blocks
        self.style_blocks = nn.ModuleList([
            StyleBlock(1024, 1024, blockIndex) for blockIndex in range(6)
        ])

        # Decoder (upsampling)
        self.decoder = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.decoderPart1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.decoderPart2 = nn.Sequential(
            # self.pad,
            nn.Conv2d(128, 3, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        )

    def forward(self, target, source):
        # Encode target face
        target = F.pad(target, pad=(3, 3, 3, 3), mode='reflect')

        target_features = self.target_encoder(target)

        # Apply style blocks
        x = target_features
        for style_block in self.style_blocks:
            x = style_block(x, source)

        # Decode
        # x = F.interpolate(x, scale_factor=2, mode='linear')
        x = F.upsample(
            x,
            scale_factor=2,  # specify the desired height and width
            mode='bilinear',  # 'linear' in 2D is called 'bilinear'
            align_corners=False  # this is typically False for ONNX compatibility
        )
        output = self.decoder(x)

        output = F.upsample(
            output,
            scale_factor=2,  # specify the desired height and width
            mode='bilinear',  # 'linear' in 2D is called 'bilinear'
            align_corners=False  # this is typically False for ONNX compatibility
        )
        output = self.decoderPart1(output)

        output = F.pad(output, pad=(3, 3, 3, 3), mode='reflect')

        output = self.decoderPart2(output)

        return (output + 1) / 2

class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blockIndex):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.style1 = nn.Linear(512, 2048)
        self.style2 = nn.Linear(512, 2048)
        self.style = [self.style1, self.style2]

        self.blockIndex = blockIndex

    def normalizeConvRMS(self, conv):
        x = conv - torch.mean(conv, dim=[2, 3], keepdim=True) # centeredConv
        squareX = x * x
        meanSquaredX = torch.mean(squareX, dim=[2, 3], keepdim=True)
        rms = torch.sqrt(meanSquaredX + 0.00000001)
        return (1 / rms) * x

    def forward(self, residual, style):
        # print(f'Forward: {self.blockIndex}')
        style1024 = []
        for index in range(2):
            style1 = self.style[index](style)
            style1 = torch.unsqueeze(style1, 2)
            style1 = torch.unsqueeze(style1, 3)
            first_half = style1[:, :1024, :, :]
            second_half = style1[:, 1024:, :, :]

            style1024.append([first_half, second_half])

        conv1 = self.normalizeConvRMS(self.conv1(F.pad(residual, pad=(1, 1, 1, 1), mode='reflect')))

        out = F.relu(conv1 * style1024[0][0] + style1024[0][1])

        out = F.pad(out, pad=(1, 1, 1, 1), mode='reflect')

        conv2 = self.normalizeConvRMS(self.conv2(out))
        out = conv2 * style1024[1][0] + style1024[1][1]

        return residual + out
