import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.fpe_fpm import SpectralAdapter, DefectPrototypeHead
from losses.fpe_fpm_losses import DefectLoss, KnownMarginLoss, BatchPrototypeLoss


class PluginSegmentationModel(nn.Module):
    def __init__(self, backbone, seg_head, feature_channels, num_classes, embed_dim=16):
        super().__init__()
        self.backbone = backbone
        self.spectral_adapter = SpectralAdapter(
            feature_channels,
            reduction=4,
            cutoff=0.22,
            enable_low_freq=True,
            enable_high_freq=True,
            enable_detail_branch=True,
            enable_gate=True,
            learnable_cutoff=True,
        )
        self.seg_head = seg_head
        self.defect_prototype_head = DefectPrototypeHead(feature_channels, embed_dim=embed_dim)
        self.num_classes = num_classes

    def forward(self, images):
        fused_feature = self.backbone(images)
        enhanced_feature = self.spectral_adapter(fused_feature)
        seg_logits = self.seg_head(enhanced_feature)
        defect_logits, embedding = self.defect_prototype_head(enhanced_feature)
        return {
            'seg_logits': seg_logits,
            'defect_logits': defect_logits,
            'embedding': embedding,
        }


def compute_training_loss(outputs, labels, ignore_label=255):
    seg_logits = outputs['seg_logits']
    defect_logits = outputs['defect_logits']
    embedding = outputs['embedding']

    if seg_logits.shape[-2:] != labels.shape[-2:]:
        seg_logits = F.interpolate(seg_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    if defect_logits.shape[-2:] != labels.shape[-2:]:
        defect_logits = F.interpolate(defect_logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
    if embedding.shape[-2:] != labels.shape[-2:]:
        embedding = F.interpolate(embedding, size=labels.shape[-2:], mode='bilinear', align_corners=False)

    base_seg_loss = F.cross_entropy(seg_logits, labels, ignore_index=ignore_label)
    defect_loss = DefectLoss(ignore_label=ignore_label)(defect_logits, labels)
    margin_loss = KnownMarginLoss(margin=0.2, ignore_label=ignore_label)(seg_logits, labels)
    prototype_loss = BatchPrototypeLoss(
        ignore_label=ignore_label,
        inter_margin=0.35,
        inter_weight=0.1,
    )(embedding, labels)

    total_loss = (
        base_seg_loss
        + 1.0 * defect_loss
        + 0.3 * margin_loss
        + 0.2 * prototype_loss
    )
    return total_loss, {
        'base_seg_loss': base_seg_loss.detach(),
        'defect_loss': defect_loss.detach(),
        'margin_loss': margin_loss.detach(),
        'prototype_loss': prototype_loss.detach(),
    }


if __name__ == '__main__':
    class ToyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(x)

    model = PluginSegmentationModel(
        backbone=ToyBackbone(),
        seg_head=nn.Conv2d(64, 5, kernel_size=1),
        feature_channels=64,
        num_classes=5,
    )
    images = torch.randn(2, 3, 128, 128)
    labels = torch.randint(0, 5, (2, 128, 128))
    outputs = model(images)
    loss, loss_items = compute_training_loss(outputs, labels)
    print(loss.item(), {k: v.item() for k, v in loss_items.items()})
