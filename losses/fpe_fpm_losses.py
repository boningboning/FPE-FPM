import torch
import torch.nn as nn
import torch.nn.functional as F


class DefectLoss(nn.Module):
    def __init__(self, ignore_label=255):
        super().__init__()
        self.ignore_label = ignore_label

    def forward(self, defect_logits, labels):
        if defect_logits is None:
            return labels.sum() * 0.0

        valid_mask = labels != self.ignore_label
        if valid_mask.sum() == 0:
            return defect_logits.sum() * 0.0

        target = (labels > 0).float()
        bce = F.binary_cross_entropy_with_logits(defect_logits.squeeze(1), target, reduction='none')
        bce = (bce * valid_mask.float()).sum() / valid_mask.float().sum()

        pred = torch.sigmoid(defect_logits.squeeze(1))
        pred = pred * valid_mask.float()
        target = target * valid_mask.float()
        intersection = (pred * target).sum()
        denominator = pred.sum() + target.sum()
        dice = 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)
        return bce + dice


class KnownMarginLoss(nn.Module):
    def __init__(self, margin=0.2, ignore_label=255):
        super().__init__()
        self.margin = margin
        self.ignore_label = ignore_label

    def forward(self, logits, labels):
        channels = logits.shape[1]
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, channels)
        labels_flat = labels.reshape(-1)
        valid_mask = (
            (labels_flat != self.ignore_label)
            & (labels_flat > 0)
            & (labels_flat < channels)
        )
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits_valid = logits_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]
        gt_logits = logits_valid.gather(1, labels_valid.unsqueeze(1)).squeeze(1)
        negative_logits = logits_valid.clone()
        negative_logits.scatter_(1, labels_valid.unsqueeze(1), float('-inf'))
        hardest_negative = negative_logits.max(dim=1).values
        return F.relu(self.margin + hardest_negative - gt_logits).mean()


class BatchPrototypeLoss(nn.Module):
    def __init__(self, ignore_label=255, inter_margin=0.35, inter_weight=0.1):
        super().__init__()
        self.ignore_label = ignore_label
        self.inter_margin = inter_margin
        self.inter_weight = inter_weight

    def forward(self, embedding, labels):
        if embedding is None:
            return labels.sum() * 0.0

        feature_dim = embedding.shape[1]
        features = embedding.permute(0, 2, 3, 1).reshape(-1, feature_dim)
        labels_flat = labels.reshape(-1)
        centers = []
        intra_losses = []

        valid_labels = labels_flat[
            (labels_flat != self.ignore_label)
            & (labels_flat > 0)
        ]
        if valid_labels.numel() == 0:
            return embedding.sum() * 0.0

        for class_id in torch.unique(valid_labels).tolist():
            class_mask = labels_flat == int(class_id)
            if class_mask.sum() == 0:
                continue
            class_feature = features[class_mask]
            center = F.normalize(class_feature.mean(dim=0, keepdim=True), dim=1).squeeze(0)
            centers.append(center)
            intra_losses.append((1.0 - F.cosine_similarity(class_feature, center.unsqueeze(0), dim=1)).mean())

        if not intra_losses:
            return embedding.sum() * 0.0

        intra_loss = torch.stack(intra_losses).mean()
        if len(centers) <= 1:
            return intra_loss

        centers = torch.stack(centers, dim=0)
        similarity = torch.matmul(centers, centers.t())
        diagonal_mask = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
        inter_similarity = similarity[~diagonal_mask]
        inter_loss = F.relu(inter_similarity - self.inter_margin).mean()
        return intra_loss + self.inter_weight * inter_loss
