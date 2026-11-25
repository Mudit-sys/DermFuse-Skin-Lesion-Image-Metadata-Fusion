"""DermFuse research demo. Model layers and transforms mirror the Kaggle notebook."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
STATS_PATH = ROOT / "meta_stats.json"
SPLIT_PATH = ROOT / "dermfuse_out" / "split.csv"
if not SPLIT_PATH.exists():
    SPLIT_PATH = ROOT.parent / "dermfuse_out" / "split.csv"
DEFAULT_DATA_DIR = Path(os.environ.get("HAM10000_DIR", ROOT / "ham_subset")).expanduser()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma (akiec)",
    "bcc": "Basal cell carcinoma (bcc)",
    "bkl": "Benign keratosis-like lesions (bkl)",
    "df": "Dermatofibroma (df)",
    "mel": "Melanoma (mel)",
    "nv": "Melanocytic nevi (nv)",
    "vasc": "Vascular lesions (vasc)",
}
DIAGNOSIS_TYPE = {
    "akiec": "pre-cancerous", "bcc": "malignant", "bkl": "benign",
    "df": "benign", "mel": "malignant", "nv": "benign", "vasc": "benign",
}
MODEL_OPTIONS = {
    "DeiT-Small, fused (default)": ["vit_fused_ce_p0.5_s42"],
    "EfficientNet-B0, fused": ["cnn_fused_ce_p0.5_s42"],
    "DeiT-Small, image-only": ["vit_image_ce_p0.5_s42"],
    "EfficientNet-B0, image-only": ["cnn_image_ce_p0.5_s42"],
    "Both fused models (ensemble)": ["vit_fused_ce_p0.5_s42", "cnn_fused_ce_p0.5_s42"],
}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EVAL_TF = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class MetaMLP(nn.Module):
    """Copied from FusionNet in DermFuse_Kaggle_v2.1.ipynb."""
    def __init__(self, d_in: int, d_hidden: int = 64, d_out: int = 32, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(d_hidden, d_out), nn.ReLU(),
        )
        self.out_dim = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionNet(nn.Module):
    """Notebook FusionNet structure, with pretrained initialization disabled."""
    def __init__(self, backbone_key: str, mode: str, meta_dim: int, n_classes: int = 7):
        super().__init__()
        assert mode in ("image", "meta", "fused")
        self.mode, self.backbone, self.meta = mode, None, None
        feature_dim = 0
        if mode in ("image", "fused"):
            model_name = {"cnn": "efficientnet_b0", "vit": "deit_small_patch16_224"}[backbone_key]
            kwargs = {"img_size": IMAGE_SIZE} if backbone_key == "vit" else {}
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0, **kwargs)
            feature_dim += self.backbone.num_features
        if mode in ("meta", "fused"):
            self.meta = MetaMLP(meta_dim)
            feature_dim += self.meta.out_dim
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, n_classes)
        )

    def features(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        values = []
        if self.backbone is not None:
            values.append(self.backbone(img))
        if self.meta is not None:
            values.append(self.meta(meta))
        return torch.cat(values, 1)

    def forward(self, img: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(img, meta))


def load_meta_stats() -> dict[str, Any]:
    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"{STATS_PATH.name} is missing. Run `python prepare_meta_stats.py --data-dir "
            "<HAM10000 folder>` first. Stats must come from the saved training split."
        )
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
    if len(stats.get("sites", [])) + 5 != 20:
        raise ValueError("Metadata schema does not have the expected 20 features.")
    if not {"age_median", "age_mean", "age_std", "sites", "sex"}.issubset(stats):
        raise ValueError("meta_stats.json is missing required training statistics.")
    return stats


META_STATS = load_meta_stats()
SITES = META_STATS["sites"]
SEXES = ["female", "male", "unknown"]


def encode_metadata(age: float | None, age_unknown: bool, sex: str, site: str) -> torch.Tensor:
    """Match the training order: age, missing flag, sex one-hot, site one-hot."""
    age_missing = age_unknown or age is None
    numeric_age = float(META_STATS["age_median"] if age_missing else age)
    values = np.zeros(20, dtype=np.float32)
    values[0] = (numeric_age - float(META_STATS["age_mean"])) / float(META_STATS["age_std"])
    values[1] = float(age_missing)
    sex = sex if sex in SEXES else "unknown"
    values[2 + SEXES.index(sex)] = 1.0
    site = site if site in SITES else "unknown"
    values[5 + SITES.index(site)] = 1.0
    return torch.from_numpy(values).unsqueeze(0).to(DEVICE)


def resize_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Upload a grayscale or RGB/RGBA dermoscopic image.")
    return np.asarray(
        Image.fromarray(np.ascontiguousarray(image.astype(np.uint8))).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
    )


def to_model_tensor(image: np.ndarray) -> torch.Tensor:
    return EVAL_TF(Image.fromarray(image)).unsqueeze(0).to(DEVICE)


def tta_views(image_tensor: torch.Tensor) -> list[torch.Tensor]:
    views = []
    for k in range(4):
        rotated = torch.rot90(image_tensor, k, dims=(2, 3))
        views.extend([rotated, torch.flip(rotated, dims=(3,))])
    return views


class ModelRegistry:
    def __init__(self) -> None:
        self.models: dict[str, tuple[FusionNet, dict[str, str]]] = {}
        for run_id in dict.fromkeys(rid for group in MODEL_OPTIONS.values() for rid in group):
            path = WEIGHTS_DIR / f"{run_id}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing model checkpoint: {path}")
            parts = run_id.split("_")
            backbone, mode = ("vit" if parts[0] == "vit" else "cnn"), parts[1]
            model = FusionNet(backbone, mode, meta_dim=20).to(DEVICE)
            state = torch.load(path, map_location=DEVICE, weights_only=True)
            incompatible = model.load_state_dict(state, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(f"Checkpoint mismatch for {run_id}: {incompatible}")
            model.eval()
            self.models[run_id] = (model, {"backbone": backbone, "mode": mode})


REGISTRY = ModelRegistry()


def predict_model(model: FusionNet, image: torch.Tensor, meta: torch.Tensor, use_tta: bool) -> np.ndarray:
    views = tta_views(image) if use_tta else [image]
    with torch.inference_mode():
        probs = [torch.softmax(model(view, meta).float(), dim=1) for view in views]
    return torch.stack(probs).mean(0)[0].cpu().numpy()


def normalize_map(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def gradcam(model: FusionNet, image: torch.Tensor, meta: torch.Tensor) -> np.ndarray:
    """Notebook Grad-CAM over the final EfficientNet feature map."""
    assert model.backbone is not None
    fmap = model.backbone.forward_features(image)
    fmap.retain_grad()
    feats = [model.backbone.forward_head(fmap)]
    if model.meta is not None:
        feats.append(model.meta(meta))
    logits = model.head(torch.cat(feats, 1))
    target = int(logits.argmax(1))
    model.zero_grad(set_to_none=True)
    logits[0, target].backward()
    weights = fmap.grad.mean((2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return normalize_map(cam.detach().cpu().numpy()[0, 0])


def attention_rollout(model: FusionNet, image: torch.Tensor, meta: torch.Tensor) -> np.ndarray:
    """Notebook attention rollout with guarded temporary fused-attention disabling."""
    assert model.backbone is not None
    maps: list[torch.Tensor] = []
    handles, changed = [], []
    blocks = model.backbone.blocks
    try:
        for block in blocks:
            attn = block.attn
            if hasattr(attn, "fused_attn"):
                changed.append((attn, bool(attn.fused_attn)))
                attn.fused_attn = False
            handles.append(attn.attn_drop.register_forward_hook(
                lambda _module, _inputs, output: maps.append(output.detach())
            ))
        with torch.no_grad():
            logits = model(image, meta)
    finally:
        for handle in handles:
            handle.remove()
        for attn, previous in changed:
            attn.fused_attn = previous
    if len(maps) != len(blocks):
        raise RuntimeError("Could not capture DeiT attention maps for rollout.")
    attention = torch.stack(maps)[:, 0].mean(1)
    tokens = attention.shape[-1]
    identity = torch.eye(tokens, device=attention.device)
    attention = attention + identity
    attention = attention / attention.sum(-1, keepdim=True)
    rollout = identity.clone()
    for layer in attention:
        rollout = layer @ rollout
    grid = int(round((tokens - 1) ** 0.5))
    if grid * grid != tokens - 1:
        raise RuntimeError(f"Unexpected DeiT token count: {tokens}")
    heat = F.interpolate(
        rollout[0, 1:].reshape(1, 1, grid, grid),
        size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False,
    )[0, 0]
    return normalize_map(heat.cpu().numpy())


def overlay_heatmap(image: np.ndarray, heat: np.ndarray) -> np.ndarray:
    colored = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return np.clip(image * 0.55 + colored * 0.45, 0, 255).astype(np.uint8)


def confidence_tier(prob: float) -> str:
    return "high" if prob >= 0.80 else "moderate" if prob >= 0.55 else "low"


def result_markdown(
    probs: np.ndarray, option: str, seconds: float, use_tta: bool,
    individual: list[tuple[str, np.ndarray]],
) -> str:
    order = np.argsort(probs)[::-1]
    top, second = int(order[0]), int(order[1])
    code, top_prob = CLASSES[top], float(probs[top])
    tier = confidence_tier(top_prob)
    lines = [
        f"### Prediction: {CLASS_NAMES[code]}",
        f"**Category:** {DIAGNOSIS_TYPE[code]}",
        f"**Confidence:** {tier} — top classes are {CLASS_NAMES[code]} ({top_prob:.1%}) "
        f"and {CLASS_NAMES[CLASSES[second]]} ({probs[second]:.1%}).",
    ]
    if tier == "low":
        lines.append("The model is hesitating between classes; specialist review is essential.")
    if len(individual) == 2:
        first_class, other_class = (int(np.argmax(item[1])) for item in individual)
        status = "agree" if first_class == other_class else "diverge"
        lines.append(
            f"**Model comparison:** EfficientNet-B0 and DeiT-Small {status} "
            f"({CLASS_NAMES[CLASSES[first_class]]} vs {CLASS_NAMES[CLASSES[other_class]]}). "
            "The displayed heatmap is from the first ensemble model (DeiT-Small)."
        )
    if any("fused" in rid for rid in MODEL_OPTIONS[option]):
        lines.append("Age, sex, and body site were used. In these experiments, metadata gave no reliable accuracy gain.")
    lines.extend([
        "The heatmap shows where the model focused in the image, not why a lesion has a medical diagnosis.",
        f"*Analysis: {seconds:.2f}s · TTA: {'on (8 views)' if use_tta else 'off'} · "
        "224×224 bilinear resize, ImageNet normalization.*",
    ])
    return "\n\n".join(lines)


def analyse(
    image: np.ndarray | None, age: float | None, age_unknown: bool, sex: str,
    site: str, option: str, use_tta: bool,
) -> tuple[str, dict[str, float] | None, np.ndarray | None]:
    if image is None:
        return "Please upload a dermoscopic image to begin.", None, None
    start = time.perf_counter()
    resized = resize_rgb(image)
    tensor = to_model_tensor(resized)
    metadata = encode_metadata(age, age_unknown, sex or "unknown", site or "unknown")
    run_ids = MODEL_OPTIONS[option]
    predictions = [
        (rid, predict_model(REGISTRY.models[rid][0], tensor, metadata, use_tta)) for rid in run_ids
    ]
    probs = np.mean([p for _rid, p in predictions], axis=0)
    first_id = run_ids[0]
    first_model, details = REGISTRY.models[first_id]
    heat = (gradcam if details["backbone"] == "cnn" else attention_rollout)(first_model, tensor, metadata)
    heatmap = overlay_heatmap(resized, heat)
    labels = {CLASS_NAMES[c]: float(probs[i]) for i, c in enumerate(CLASSES)}
    text = result_markdown(probs, option, time.perf_counter() - start, use_tta, predictions)
    return text, labels, heatmap


def find_examples() -> list[list[Any]]:
    """Return up to six saved-split test images; local images are never copied."""
    root = DEFAULT_DATA_DIR
    if not root.exists() or not SPLIT_PATH.exists():
        return []
    meta_path = root / "HAM10000_metadata.csv"
    if not meta_path.is_file():
        return []
    try:
        metadata = pd.read_csv(meta_path)
        split = pd.read_csv(SPLIT_PATH)
        test = metadata.merge(split[["image_id", "split"]], on="image_id", how="inner")
        test = test[test["split"] == "test"]
        images = {p.stem: p for p in root.rglob("*.jpg")}
        examples = []
        for code in CLASSES:
            candidates = test[test["dx"] == code].sort_values("image_id")
            if candidates.empty:
                continue
            row = candidates.iloc[0]
            path = images.get(str(row["image_id"]))
            if path is None:
                continue
            missing_age = pd.isna(row.get("age"))
            raw_sex = row.get("sex")
            raw_site = row.get("localization")
            sex = "unknown" if pd.isna(raw_sex) or str(raw_sex).lower() not in SEXES else str(raw_sex).lower()
            site = "unknown" if pd.isna(raw_site) or str(raw_site).lower() not in SITES else str(raw_site).lower()
            examples.append([
                str(path), None if missing_age else float(row["age"]), bool(missing_age), sex, site,
                "DeiT-Small, fused (default)", True,
            ])
        return examples[:6]
    except Exception as exc:
        print(f"Skipping local examples: {exc}")
        return []


WARNING = (
    "**Research tool, not a medical device.** Trained only on dermoscopic images (HAM10000). "
    "A phone photo is outside its domain and results are unreliable. Probabilities are **not calibrated** "
    "and the model can be confidently wrong. Melanoma sensitivity on the test set is only about 0.5, "
    "so a non-melanoma result is not reassuring. No medical decision should be made from this tool."
)
SUBTITLE = (
    "EfficientNet-B0 and DeiT-Small, with image-only and image-plus-metadata modes. "
    "Mean test macro-F1 was about 0.65-0.68 across three seeds on the HAM10000 lesion-grouped split."
)
CSS = """
:root { --df-ink:#182a34; --df-muted:#54666e; --df-paper:#f3f6f5; --df-line:#d6e0df; --df-accent:#176b69; }
body,.gradio-container { background:var(--df-paper)!important; color:var(--df-ink)!important; }
.gradio-container { max-width:1180px!important; }
#df-hero { padding:1.4rem 0 .65rem; }
#df-hero h1 { color:var(--df-ink); font-size:clamp(2rem,4vw,3rem); letter-spacing:-.035em; margin-bottom:.35rem; }
#df-hero p { color:var(--df-muted); max-width:760px; line-height:1.55; }
#df-warning { border:1px solid #d5a2a0; border-left:5px solid #9e3836; background:#fff6f4!important; border-radius:8px; padding:.9rem 1rem; }
#df-warning, #df-warning p, #df-warning strong { color:#492625!important; }
.panel { border:1px solid var(--df-line)!important; border-radius:10px!important; background:#fff!important; }
button.primary { background:var(--df-accent)!important; border-color:var(--df-accent)!important; }
button:focus-visible,input:focus-visible,select:focus-visible { outline:3px solid #70aaa6!important; outline-offset:2px!important; }
@media(max-width:700px) { #df-hero { padding-top:.6rem; } }
"""

with gr.Blocks(title="Skin lesion classification: research demo", css=CSS) as demo:
    gr.Markdown("# Skin lesion classification: research demo\n\n" + SUBTITLE, elem_id="df-hero")
    gr.Markdown(WARNING, elem_id="df-warning")
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes=["panel"]):
            image_input = gr.Image(type="numpy", image_mode="RGB", label="Dermoscopic image", sources=["upload"])
            with gr.Row():
                age_input = gr.Number(label="Age", minimum=0, maximum=120, precision=0)
                age_unknown = gr.Checkbox(label="Age unknown", value=False)
            sex_input = gr.Dropdown(choices=SEXES, value="unknown", label="Sex")
            site_input = gr.Dropdown(choices=SITES, value="unknown", label="Body site")
            model_input = gr.Dropdown(choices=list(MODEL_OPTIONS), value="DeiT-Small, fused (default)", label="Model")
            tta_input = gr.Checkbox(value=True, label="TTA (8 views: slower, usually a little more accurate)")
            analyse_button = gr.Button("Analyse", variant="primary", size="lg")
            examples = find_examples()
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=[image_input, age_input, age_unknown, sex_input, site_input, model_input, tta_input],
                    label="Test-split examples (one per class)",
                )
        with gr.Column(scale=1, elem_classes=["panel"]):
            explanation_output = gr.Markdown("Upload an image and select **Analyse**.")
            label_output = gr.Label(num_top_classes=7, label="Model probabilities (uncalibrated)")
            heatmap_output = gr.Image(label="Grad-CAM / attention rollout: regions that influenced the prediction")
    analyse_button.click(
        fn=analyse,
        inputs=[image_input, age_input, age_unknown, sex_input, site_input, model_input, tta_input],
        outputs=[explanation_output, label_output, heatmap_output],
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", share=False)
