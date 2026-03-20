# Fusion Agent Reference

## Bayesian Fusion (Current)

See agent catalog in `agents.md` for implementation.

Default weights (empirically set, to be tuned on FaceForensics++):
- geometry: 0.15
- frequency: 0.25
- texture: 0.20
- biological: 0.15
- vlm: 0.25

## Learned Fusion (Phase 2)

### Training Data
- FaceForensics++ (c23 compression): ~5000 real, ~5000 fake images
- Extract agent score vectors for each image
- Train/val/test split: 70/15/15

### Model
```python
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
```

### Training Script
- To be added when all feature agents are non-stub
