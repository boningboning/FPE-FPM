# FPE-FPM: Frequency-Prototype Enhancement based Feature Post-processing Module

This folder contains only the proposed FPE-FPM innovation code, without the PIDNet baseline implementation.

## Files

```text
modules/fpe_fpm.py                      # SpectralAdapter + DefectPrototypeHead
losses/fpe_fpm_losses.py                # DefectLoss + KnownMarginLoss + BatchPrototypeLoss
examples/fpe_fpm_integration_example.py # minimal integration example for any segmentation backbone
configs/fpe_fpm.yaml                    # hyperparameters used in the final model
```

## Main components

- `SpectralAdapter`: frequency-detail enhancement module for a final fused feature map.
- `DefectPrototypeHead`: defect-aware binary head and pixel embedding head.
- `DefectLoss`: BCE + Dice loss for defect/non-defect supervision.
- `KnownMarginLoss`: foreground class margin regularization on segmentation logits.
- `BatchPrototypeLoss`: batch-wise class prototype compactness and separation loss.

## Default loss weights

```text
DefectLoss: 1.0
KnownMarginLoss: 0.3
BatchPrototypeLoss: 0.2
```

## Minimal usage

```python
from modules.fpe_fpm import SpectralAdapter, DefectPrototypeHead

spectral_adapter = SpectralAdapter(channels=256)
defect_head = DefectPrototypeHead(channels=256, embed_dim=16)

f_enhanced = spectral_adapter(fused_feature)
defect_logits, embedding = defect_head(f_enhanced)
seg_logits = segmentation_head(f_enhanced)
```

See `examples/fpe_fpm_integration_example.py` for a runnable toy example.
