"""استخراج الميزات (Feature Extraction) بنموذج ResNet50 جاهز.

كل صورة -> متجه 2048 رقم (بصمة رقمية) بعد تطبيعه (L2) عشان نقارن بـ cosine.
"""
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .config import IMG_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_image(path_or_file) -> Image.Image:
    return Image.open(path_or_file).convert("RGB")


def to_tensor(img: Image.Image) -> torch.Tensor:
    return _preprocess(img)


class FeatureExtractor(nn.Module):
    """ResNet50 بدون طبقة التصنيف الأخيرة. بيرجّع كمان خرائط layer4 للتفسير."""

    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        base = models.resnet50(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                  base.layer1, base.layer2, base.layer3)
        self.layer4 = base.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.eval()
        self.to(DEVICE)

    def forward(self, x, return_maps=False):
        maps = self.layer4(self.stem(x))              # (B, 2048, 7, 7)
        emb = self.pool(maps).flatten(1)              # (B, 2048)
        emb = nn.functional.normalize(emb, dim=1)
        return (emb, maps) if return_maps else emb


_extractor = None


def get_extractor() -> FeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


@torch.no_grad()
def embed_batch(tensors) -> np.ndarray:
    model = get_extractor()
    x = torch.stack(tensors).to(DEVICE)
    return model(x).cpu().numpy().astype("float32")


@torch.no_grad()
def embed_image(img: Image.Image) -> np.ndarray:
    return embed_batch([to_tensor(img)])[0]
