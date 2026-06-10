import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SpectralAdapter(nn.Module):
    def __init__(
        self,
        channels,
        reduction=4,
        cutoff=0.22,
        enable_low_freq=True,
        enable_high_freq=True,
        enable_detail_branch=True,
        enable_gate=True,
        learnable_cutoff=True,
    ):
        super().__init__()
        hidden_channels = max(channels // reduction, 16)
        self.enable_low_freq = enable_low_freq
        self.enable_high_freq = enable_high_freq
        self.enable_detail_branch = enable_detail_branch
        self.enable_gate = enable_gate
        self.learnable_cutoff = learnable_cutoff
        self.reduce = ConvBNReLU(channels, hidden_channels, kernel_size=1)
        self.detail = None
        if self.enable_detail_branch:
            self.detail = ConvBNReLU(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
            )
        branch_count = int(self.enable_low_freq) + int(self.enable_high_freq) + int(self.enable_detail_branch)
        self.fuse = None
        if branch_count > 0:
            self.fuse = nn.Sequential(
                nn.Conv2d(hidden_channels * branch_count, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
            )
        gate_hidden = max(channels // reduction, 16)
        self.gate = None
        if self.enable_gate:
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, gate_hidden, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_hidden, channels, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
        cutoff_tensor = torch.tensor(float(cutoff))
        if self.learnable_cutoff:
            self.cutoff = nn.Parameter(cutoff_tensor)
        else:
            self.register_buffer('cutoff', cutoff_tensor)
        self.activation = nn.ReLU(inplace=True)

    def _radial_mask(self, height, width, device, dtype):
        cutoff = torch.clamp(self.cutoff, min=0.05, max=0.45)
        y = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
        x = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt(xx * xx + yy * yy)
        low_mask = torch.sigmoid((cutoff - radius) * 12.0)
        return low_mask.unsqueeze(0).unsqueeze(0)

    def _frequency_split(self, feature):
        freq = torch.fft.fftshift(torch.fft.fft2(feature, norm='ortho'))
        low_mask = self._radial_mask(feature.shape[-2], feature.shape[-1], feature.device, feature.dtype)
        high_mask = 1.0 - low_mask
        low_feature = torch.fft.ifft2(torch.fft.ifftshift(freq * low_mask), norm='ortho').real
        high_feature = torch.fft.ifft2(torch.fft.ifftshift(freq * high_mask), norm='ortho').real
        return low_feature, high_feature

    def forward(self, feature):
        reduced = self.reduce(feature)
        branches = []
        if self.enable_low_freq or self.enable_high_freq:
            low_feature, high_feature = self._frequency_split(reduced.float())
            if self.enable_low_freq:
                branches.append(low_feature.to(reduced.dtype))
            if self.enable_high_freq:
                branches.append(high_feature.to(reduced.dtype))
        if self.enable_detail_branch:
            detail_feature = self.detail(reduced - F.avg_pool2d(reduced, kernel_size=3, stride=1, padding=1))
            branches.append(detail_feature)
        if not branches:
            return feature
        fused = self.fuse(torch.cat(branches, dim=1))
        if self.gate is not None:
            fused = self.gate(feature) * fused
        return self.activation(feature + fused)


class DefectPrototypeHead(nn.Module):
    def __init__(self, channels, embed_dim=16):
        super().__init__()
        hidden_channels = max(channels // 2, 32)
        branch_channels = max(hidden_channels // 2, 16)
        self.defect_head = nn.Sequential(
            ConvBNReLU(channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )
        self.embed_head = nn.Sequential(
            ConvBNReLU(channels, hidden_channels, kernel_size=3, padding=1),
            ConvBNReLU(hidden_channels, branch_channels, kernel_size=3, padding=1),
            nn.Conv2d(branch_channels, embed_dim, kernel_size=1, bias=True),
        )

    def forward(self, feature):
        defect_logits = self.defect_head(feature)
        embedding = F.normalize(self.embed_head(feature), dim=1)
        return defect_logits, embedding
