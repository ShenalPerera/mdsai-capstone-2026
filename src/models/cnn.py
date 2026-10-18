"""
Arm 2, CNN: ImageNet-pretrained EfficientNet-B0 with a shared encoder and
two heads, trained on the 224px letterboxed image cache.

    image 3x224x224 -> EfficientNet-B0 features -> average pool
      -> 1280-d embedding -> dropout 0.2 -> flaw head    (1280 -> 6)
                                         -> UNREC head   (1280 -> 1)

The model returns 7 logits in OUTPUTS order (BLR BRT DRK FRM OBS ROT UNREC),
the same columns as the classical arm, so both are scored by
src/models/metrics.py. The 1280-d embedding is exposed for the hybrid arm
(Stage E).

Two training regimes:
- frozen: backbone weights fixed and kept in eval mode (BatchNorm running
  statistics stay at their ImageNet values); only the heads learn.
- finetune: everything learns; the backbone at a 10x lower learning rate
  than the randomly initialised heads, so the pretrained features are
  adjusted rather than overwritten.
"""
import torch
from torch import nn
from torchvision import transforms as T
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from src.data.splits import MODELLED_FLAWS
from src.models.classical import OUTPUTS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBEDDING_DIM = 1280
DROPOUT = 0.2  # torchvision's EfficientNet-B0 classifier default


def image_transform(train):
    """Cached images are already 224x224, so no resize or crop. Training
    adds a horizontal flip only (docs/decisions.md): crops change frame
    content (FRM), rotations change orientation (ROT), colour jitter changes
    exposure (DRK/BRT). The padding colour normalises to ~0."""
    steps = [T.RandomHorizontalFlip(0.5)] if train else []
    return T.Compose(steps + [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class QualityNet(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b0(weights=weights, progress=False)
        self.backbone = net.features
        self.pool = net.avgpool
        self.dropout = nn.Dropout(DROPOUT)
        self.flaw_head = nn.Linear(EMBEDDING_DIM, len(MODELLED_FLAWS))
        self.unrec_head = nn.Linear(EMBEDDING_DIM, 1)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        """A frozen backbone stays in eval mode even while the heads train,
        so BatchNorm does not update its running statistics."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def embed(self, x):
        return torch.flatten(self.pool(self.backbone(x)), 1)

    def forward(self, x, return_embedding=False):
        embedding = self.embed(x)
        h = self.dropout(embedding)
        logits = torch.cat([self.flaw_head(h), self.unrec_head(h)], dim=1)
        return (logits, embedding) if return_embedding else logits

    def param_groups(self, lr_head, lr_backbone):
        """AdamW parameter groups: heads at lr_head, backbone at lr_backbone
        (left out entirely when frozen)."""
        heads = list(self.flaw_head.parameters()) + list(self.unrec_head.parameters())
        groups = [{"params": heads, "lr": lr_head, "name": "heads"}]
        if not self.freeze_backbone:
            groups.append({"params": list(self.backbone.parameters()), "lr": lr_backbone,
                           "name": "backbone"})
        return groups


def pos_weights(targets):
    """BCE pos_weight per output = negatives / positives on the given
    targets (the training partition): the BCE counterpart of the classical
    arm's class_weight="balanced". targets: 0/1 DataFrame with OUTPUTS columns."""
    pos = targets[OUTPUTS].sum()
    return torch.tensor(((len(targets) - pos) / pos).to_numpy(), dtype=torch.float32)
