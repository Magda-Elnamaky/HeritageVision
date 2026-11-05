"""التفسير (Explainable AI): ليه النظام شاف الصورتين دول متشابهين؟

بنحسب Grad-CAM على similarity بين صورة الاستعلام والنتيجة:
الأجزاء الحمرا في صورة الاستعلام هي اللي ساهمت أكتر في التشابه.
"""
import numpy as np
import torch
from PIL import Image

from .features import DEVICE, get_extractor, to_tensor


def similarity_heatmap(query_img: Image.Image, target_emb: np.ndarray) -> np.ndarray:
    """ترجّع heatmap (7x7) قيمها من 0 لـ 1 على صورة الاستعلام."""
    model = get_extractor()
    x = to_tensor(query_img).unsqueeze(0).to(DEVICE)
    target = torch.from_numpy(target_emb).to(DEVICE)

    with torch.enable_grad():
        maps = model.layer4(model.stem(x))
        maps.retain_grad()
        emb = torch.nn.functional.normalize(model.pool(maps).flatten(1), dim=1)[0]
        score = torch.dot(emb, target)
        model.zero_grad()
        score.backward()

    weights = maps.grad.mean(dim=(2, 3), keepdim=True)         # (1, C, 1, 1)
    cam = torch.relu((weights * maps).sum(dim=1))[0]             # (7, 7)
    cam = cam.detach().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """يركّب الـ heatmap على الصورة (بعد نفس القص المركزي بتاع المعالجة)."""
    import matplotlib.cm as cm

    w, h = img.size
    s = int(min(w, h) * 224 / 256)  # نفس Resize(256) ثم CenterCrop(224)
    left, top = (w - s) // 2, (h - s) // 2
    crop = img.crop((left, top, left + s, top + s)).resize((224, 224))
    cam_img = Image.fromarray((cam * 255).astype("uint8")).resize((224, 224), Image.BICUBIC)
    heat = (cm.jet(np.asarray(cam_img) / 255.0)[..., :3] * 255).astype("uint8")
    blended = (np.asarray(crop) * (1 - alpha) + heat * alpha).astype("uint8")
    return Image.fromarray(blended)
