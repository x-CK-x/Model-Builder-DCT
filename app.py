############################ multi-device batch, Grad-CAM, live progress ############################

#  • selectable compute resources (CPU cores + individual GPU ids)
#  • thread-safe queue with worker pool – each checked device grabs images as soon as it is free
#  • live progress bar in Gradio (auto-updates via generator)
#  • Grad-CAM restored in Single-Image tab (click a predicted tag)
#  • model replica cached per device so GPUs work in parallel without contention
#####################################################################################################

from __future__ import annotations

import copy
import os, re, queue, threading
from pathlib import Path
from typing import Tuple, Dict, List, Generator

from PIL import Image
import numpy as np, matplotlib.cm as cm
import msgspec, torch
from torchvision.transforms import transforms, InterpolationMode
import torchvision.transforms.functional as TF
import timm, safetensors.torch, gradio as gr
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import (
    LlavaForConditionalGeneration,
    TextIteratorStreamer,
    AutoProcessor,
)
import json, math
import numpy as np
from openrouter_tab import add_openrouter_tab
from clean_tags_tab import add_clean_tags_tabs
from collections import defaultdict
from pathlib import Path

out_dir = None

# ╭───────────────────────── Device helpers ─────────────────────────╮
def cuda_devices() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]

GPU_LABELS = cuda_devices()
MODEL_ID      = None
NUM_CLASSES   = None
WEIGHT_PATH   = None
MODEL_SLUG    = None

# ╭────────────────────────────── Local model cache ─────────────────────────────╮
BASE_DIR   = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"               # all checkpoints & tag files live here
MODELS_DIR.mkdir(exist_ok=True)
# ╰───────────────────────────────────────────────────────────────────────────────╯
# ╭────────────────────── Dynamic multi-model registry ─────────────────────────╮
REG = json.load(open(BASE_DIR / "model_registry.json", "r"))

class GatedHead(torch.nn.Module):
    def __init__(self, in_feats, n_cls):
        super().__init__()
        self.linear = torch.nn.Linear(in_feats, n_cls * 2)
        self.act, self.gate = torch.nn.Sigmoid(), torch.nn.Sigmoid()
        self.c = n_cls
    def forward(self, x):
        x = self.linear(x)
        return self.act(x[:, : self.c]) * self.gate(x[:, self.c:])

_model_cache: Dict[str, timm.models.VisionTransformer] = {}

# ────────────── Caption model setup ──────────────
CAPTION_REPO = "fancyfeast/llama-joycaption-beta-one-hf-llava"
CAPTION_CACHE = Path.home() / ".cache" / "joycaption"
CAPTION_CACHE.mkdir(parents=True, exist_ok=True)

_caption_cache: dict[str, LlavaForConditionalGeneration] = {}

def load_caption_model(device: torch.device) -> LlavaForConditionalGeneration:
    key = str(device)
    if key in _caption_cache:
        return _caption_cache[key]
    processor = AutoProcessor.from_pretrained(CAPTION_REPO, cache_dir=CAPTION_CACHE)
    model = LlavaForConditionalGeneration.from_pretrained(
        CAPTION_REPO,
        torch_dtype=torch.bfloat16,
        device_map={"": device.index if device.type == "cuda" else "cpu"},
        cache_dir=CAPTION_CACHE,
    )
    model.processor = processor
    model.eval()
    _caption_cache[key] = model
    return model

CAPTION_TYPE_MAP = {
    "Descriptive": [
        "Write a detailed description for this image.",
        "Write a detailed description for this image in {word_count} words or less.",
        "Write a {length} detailed description for this image.",
    ],
    "Descriptive (Casual)": [
        "Write a descriptive caption for this image in a casual tone.",
        "Write a descriptive caption for this image in a casual tone within {word_count} words.",
        "Write a {length} descriptive caption for this image in a casual tone.",
    ],
    "Straightforward": [
        "Write a straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a straightforward caption for this image within {word_count} words. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
        "Write a {length} straightforward caption for this image. Begin with the main subject and medium. Mention pivotal elements—people, objects, scenery—using confident, definite language. Focus on concrete details like color, shape, texture, and spatial relationships. Show how elements interact. Omit mood and speculative wording. If text is present, quote it exactly. Note any watermarks, signatures, or compression artifacts. Never mention what's absent, resolution, or unobservable details. Vary your sentence structure and keep the description concise, without starting with “This image is…” or similar phrasing.",
    ],
    "Stable Diffusion Prompt": [
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
        "Output a stable diffusion prompt that is indistinguishable from a real stable diffusion prompt. {word_count} words or less.",
        "Output a {length} stable diffusion prompt that is indistinguishable from a real stable diffusion prompt.",
    ],
    "MidJourney": [
        "Write a MidJourney prompt for this image.",
        "Write a MidJourney prompt for this image within {word_count} words.",
        "Write a {length} MidJourney prompt for this image.",
    ],
    "Danbooru tag list": [
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {word_count} words or less.",
        "Generate only comma-separated Danbooru tags (lowercase_underscores). Strict order: `artist:`, `copyright:`, `character:`, `meta:`, then general tags. Include counts (1girl), appearance, clothing, accessories, pose, expression, actions, background. Use precise Danbooru syntax. No extra text. {length} length.",
    ],
    "e621 tag list": [
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
        "Write a comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of e621 tags in alphabetical order for this image. Start with the artist, copyright, character, species, meta, and lore tags (if any), prefixed by 'artist:', 'copyright:', 'character:', 'species:', 'meta:', and 'lore:'. Then all the general tags.",
    ],
    "Rule34 tag list": [
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
        "Write a comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags. Keep it under {word_count} words.",
        "Write a {length} comma-separated list of rule34 tags in alphabetical order for this image. Start with the artist, copyright, character, and meta tags (if any), prefixed by 'artist:', 'copyright:', 'character:', and 'meta:'. Then all the general tags.",
    ],
    "Booru-like tag list": [
        "Write a list of Booru-like tags for this image.",
        "Write a list of Booru-like tags for this image within {word_count} words.",
        "Write a {length} list of Booru-like tags for this image.",
    ],
    "Art Critic": [
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
        "Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
    ],
    "Product Listing": [
        "Write a caption for this image as though it were a product listing.",
        "Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
        "Write a {length} caption for this image as though it were a product listing.",
    ],
    "Social Media Post": [
        "Write a caption for this image as if it were being used for a social media post.",
        "Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
        "Write a {length} caption for this image as if it were being used for a social media post.",
    ],
}

NAME_OPTION = "If there is a person/character in the image you must refer to them as {name}."


def build_prompt(caption_type: str, caption_length: str | int, extra_options: list[str], name_input: str) -> str:
    if caption_length == "any":
        idx = 0
    elif isinstance(caption_length, str) and caption_length.isdigit():
        idx = 1
    else:
        idx = 2
    prompt = CAPTION_TYPE_MAP[caption_type][idx]
    if extra_options:
        prompt += " " + " ".join(extra_options)
    return prompt.format(name=name_input or "{NAME}", length=caption_length, word_count=caption_length)


def toggle_name_box(selected_options: list[str]):
    return gr.update(visible=NAME_OPTION in selected_options)

def caption_once(img: Image.Image, prompt: str, temperature: float, top_p: float, max_new_tokens: int, device: torch.device) -> str:
    model = load_caption_model(device)
    processor = model.processor
    convo = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.strip()},
    ]
    convo_str = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[convo_str], images=[img], return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        use_cache=True,
    )
    return processor.batch_decode(out[:, inputs["input_ids"].shape[-1]:])[0].strip()


def caption_single(img: Image.Image, caption_type: str, caption_length: str | int,
                   extra_opts: list[str], name_field: str,
                   temperature: float, top_p: float, max_new_tokens: int,
                   devices: list[str]):
    if img is None:
        return ""
    device = _pick_device(devices)
    prompt = build_prompt(caption_type, caption_length, extra_opts, name_field)
    return caption_once(img, prompt, temperature, top_p, max_new_tokens, device)

def local_path(spec: dict, fname: str) -> Path:
    return MODELS_DIR / spec["subfolder"] / fname

def load_model(key: str, device: torch.device, progress: gr.Progress | None = None):
    """
    Returns a VisionTransformer for `key`, cached per device.
    If the model files are missing they are downloaded with a tiny
    progress bar so the user can see what's happening.
    """
    spec = REG[key]
    cache_k = f"{key}|{device}"
    if cache_k in _model_cache:
        return _model_cache[cache_k]

    # target path under ./models/<subfolder>/<filename>
    # 1️⃣ choose *root* dir once (no nested subfolder here)
    ckpt_root = MODELS_DIR  # .../models
    ckpt_path = ckpt_root / spec["subfolder"] / spec["filename"]

    # 2️⃣ download: keep subfolder param, but local_dir = ckpt_root
    if not ckpt_path.exists():
        (ckpt_root / spec["subfolder"]).mkdir(parents=True, exist_ok=True)
        tracker = progress or gr.Progress(track_tqdm=True)
        tracker(0, desc=f"Downloading {key} …", total=1, unit="file")

        hf_hub_download(
            repo_id=spec["repo"],
            subfolder=spec["subfolder"],  # <- REMOTE path
            filename=spec["filename"],
            local_dir=ckpt_root,  # <- LOCAL root (models/)
        )
        tracker(1)

    if spec.get("backend", "pytorch") == "onnx":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.type == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(ckpt_path), providers=providers)
        _model_cache[cache_k] = session
        return session
    else:
        m = timm.create_model(spec["timm_id"],
                              pretrained=False,
                              num_classes=spec["num_classes"]).to(device)
        if spec.get("head_type", "gated") == "gated":
            m.head = GatedHead(min(m.head.weight.shape), spec["num_classes"])
        else:
            m.head = torch.nn.Linear(min(m.head.weight.shape), spec["num_classes"])

        safetensors.torch.load_model(m, str(ckpt_path), strict=False)

        m.to(device)
        m.eval()
        _model_cache[cache_k] = m
        return m
# ╰──────────────────────────────────────────────────────────────────────────────╯

# ╭──────────── Image transforms ──────────────╮
class Fit(torch.nn.Module):
    def __init__(self, bounds, *, interpolation=InterpolationMode.LANCZOS, grow=True, pad=None):
        super().__init__()
        self.b = (bounds, bounds) if isinstance(bounds, int) else bounds
        self.inter = interpolation
        self.grow = grow
        self.pad = pad

    def forward(self, img: Image.Image):
        w, h = img.size
        bw, bh = self.b
        s = min(bw / w, bh / h)
        s = min(s, 1.0) if not self.grow else s
        if s != 1:
            img = TF.resize(img, (round(h * s), round(w * s)), self.inter)
        if self.pad is None:
            return img
        ph, pw = bh - img.size[1], bw - img.size[0]
        return TF.pad(img, (pw // 2, ph // 2, pw - pw // 2, ph - ph // 2), self.pad)


class CompositeAlpha(torch.nn.Module):
    def __init__(self, background: tuple[float, float, float] | float):
        super().__init__()
        bg = (background, background, background) if isinstance(background, float) else background
        self.register_buffer("bg", torch.tensor(bg).view(3, 1, 1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if img.shape[-3] == 3:
            return img
        a = img[..., 3:, :, :]
        return img[..., :3, :, :] * a + self.bg * (1 - a)


TRANSFORM = transforms.Compose(
    [
        Fit((384, 384)),
        transforms.ToTensor(),
        CompositeAlpha(0.5),
        transforms.Normalize([0.5] * 3, [0.5] * 3, inplace=True),
        transforms.CenterCrop((384, 384)),
    ]
)
# ╰────────────────────────────────────────────╯

# ╭──────────── Tags & helpers ─────────────╮
with open("tagger_tags.json", "rb") as f:
    TAGS_R: Dict[str, int] = msgspec.json.decode(f.read(), type=Dict[str, int])
TAGS = {k.replace("_", " "): v for k, v in TAGS_R.items()}
ALLOWED = list(TAGS.keys())

def classify_tensor(t: torch.Tensor, m, thr, head_type="gated", backend="pytorch"):
    with torch.no_grad():
        if backend == "onnx":
            input_name = m.get_inputs()[0].name
            output_name = m.get_outputs()[0].name
            out = m.run([output_name], {input_name: t.cpu().numpy()})[0]
            logits = torch.from_numpy(out)[0]
            probits = torch.sigmoid(logits)
        else:
            logits = m(t)[0]
            probits = torch.sigmoid(logits) if head_type == "linear" else logits
        vals, idxs = probits.cpu().topk(250)
    sc={ALLOWED[i.item()]:v.item() for i,v in zip(idxs,vals)}
    filt={k:v for k,v in sc.items() if v>thr}
    return ", ".join(filt.keys()), sc
# ╰──────────────────────────────────────────╯

# ╭──────────── Grad-CAM ─────────────╮
def _best_grid(n: int) -> tuple[int, int]:
    """Return (h, w) such that h × w = n and h is closest to √n."""
    root = int(math.sqrt(n))
    for h in range(root, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n

def _patch_grid_from_token_count(P: int, img_h: int, img_w: int) -> tuple[int, int]:
    """
    Infer (h, w) such that
      • h * w == P
      • h divides img_h  AND  w divides img_w     (keeps patches aligned)
      • |h - sqrt(P)| is minimal  (looks as square as possible)
    """
    best = None
    root = int(math.sqrt(P))
    for h in range(1, root + 1):
        if P % h:
            continue
        w = P // h
        if img_h % h or img_w % w:           # must tile the padded square
            continue
        best = (h, w)
    return best if best else (root, P // root)


# ───────────────── Patch-grid helper (robust) ─────────────────
def _patch_grid(model, P: int) -> tuple[int, int]:
    """
    Return (H, W) patch grid for ViT / SigLIP / DeiT / etc.

    Priority:
    1.  Use (img_size // proj.stride)              ← handles rectangular strides
    2.  Use model.patch_embed.grid_size            ← only if it matches P
    3.  Derive a factor pair of P (fallback)
    """
    pe = getattr(model, "patch_embed", None)
    if pe is not None:
        # --- 1. derive from conv stride (works for SigLIP & friends)
        img_sz   = getattr(pe, "img_size", None)
        proj     = getattr(pe, "proj", None)
        if img_sz is not None and proj is not None and hasattr(proj, "stride"):
            ih, iw = (img_sz if isinstance(img_sz, (tuple, list)) else (img_sz, img_sz))
            sh, sw = proj.stride if isinstance(proj.stride, tuple) else (proj.stride, proj.stride)
            h, w = ih // sh, iw // sw
            if h * w == P:
                return h, w

        # --- 2. trust grid_size *only* if area matches
        gs = getattr(pe, "grid_size", None)
        if gs is not None and int(gs[0]) * int(gs[1]) == P:
            return int(gs[0]), int(gs[1])

    # --- 3. generic square-ish fallback
    root = int(math.sqrt(P))
    for h in range(root, 0, -1):
        if P % h == 0:
            return h, P // h
    return 1, P

# ─────────────────────────  DEBUG helper  ─────────────────────────
def debug_show_patch_coords(model, *, img_size=384, square=216):
    """
    Returns a PIL image in which every patch is coloured by its (row, col) id.
    Lets you see whether reshape(h, w) maps patches to the right pixels.
    """
    P = getattr(model.patch_embed, "num_patches", None) or model.patch_embed.num_patches
    h, w = _patch_grid_from_token_count(P, img_size, img_size)
    grid = np.arange(P, dtype=np.float32).reshape(h, w)

    # normalise 0-1 for a colourmap
    grid -= grid.min();  grid /= grid.max()

    rgb = (cm.viridis(grid)[..., :3] * 255).astype("uint8")
    im  = Image.fromarray(rgb, mode="RGB")

    im = im.resize((square, square), Image.NEAREST)     # coarse upscale (blocky)
    im = im.resize((img_size, img_size), Image.BICUBIC) # pad back to square
    return im


# 1️⃣ get_cam_array  ───────────────────────────────────────────
def get_cam_array(img: Image.Image, tag: str, model) -> np.ndarray:
    dev = next(model.parameters()).device
    idx = TAGS[tag]

    t = TRANSFORM(img.convert("RGBA")).unsqueeze(0).to(dev).requires_grad_(True)

    acts, grads = {}, {}
    def fwd(_, __, o): acts["v"] = o
    def bwd(_, gi, go): grads["v"] = go[0]

    h1 = model.norm.register_forward_hook(fwd)
    h2 = model.norm.register_full_backward_hook(bwd)

    try:
        model.zero_grad(set_to_none=True)
        model(t)[0, idx].backward()

        a = acts["v"].squeeze(0)
        g = grads["v"].squeeze(0)
        w = g.mean(1)
        cam_1d = torch.relu((a * w[:, None]).sum(0)).detach().cpu().numpy()

        h, w_ = _patch_grid_from_token_count(cam_1d.size, 384, 384)
        print(f"debug print right after the _patch_grid_from_token_count call, but before the return cam_1d.reshape(h, w_), a   call:")
        print("TOKEN COUNT:", cam_1d.size, "→ grid", h, "×", w_)

        return cam_1d.reshape(h, w_), a
    finally:
        h1.remove(); h2.remove()

def _rgba_to_heat(rgba: Image.Image) -> np.ndarray:
    """
    Convert a CAM RGBA overlay (216×216, uint8) into a float32 heat array
    shaped (H, W, 4) where the last dim is premultiplied RGB + α channel.

    • RGB channels are divided by 255 and multiplied by α/255
    • α is kept as α/255
    """
    arr = np.asarray(rgba, dtype=np.float32) / 255.0      # H,W,4
    rgb, a = arr[..., :3], arr[..., 3:4]
    return np.concatenate([rgb * a, a], axis=-1)          # premult RGB, keep α


def create_cam_visualization_pil(image_pil: Image.Image, cam: np.ndarray,
                                 *, alpha: float = 0.6, vis_threshold: float = 0.2) -> Image.Image:
    """Exact port of the HF‑demo visualiser."""
    if cam is None:
        return image_pil

    w, h = image_pil.size
    size = max(w, h)

    cam -= cam.min()
    cam /= cam.max() + 1e-8

    colormap = cm.get_cmap("inferno")
    cam_rgb = colormap(cam)[..., :3]  # (H,W,3) float64 0‑1

    cam_alpha = (cam >= vis_threshold).astype(np.float32) * alpha
    cam_rgba = np.dstack((cam_rgb, cam_alpha))

    cam_pil = Image.fromarray((cam_rgba * 255).astype(np.uint8), mode="RGBA")
    cam_pil = cam_pil.resize((216, 216), Image.NEAREST)          # keep blocky
    cam_pil = cam_pil.resize((size, size), Image.BICUBIC)
    cam_pil = transforms.CenterCrop((h, w))(cam_pil)

    return Image.alpha_composite(image_pil.convert("RGBA"), cam_pil)

def _cam_to_overlay(cam: np.ndarray, img: Image.Image,
                    *, alpha: float, thr: float) -> Image.Image:
    """
    cam : 2-D float array (already normalised 0-1)
    img : original PIL image   (for final crop size)
    Returns: PIL RGBA image sized exactly like *img*.
    """
    cm_rgb     = (cm.inferno(cam)[..., :3] * 255).astype("uint8")
    alpha_mask = ((cam >= thr) * alpha * 255).astype("uint8")
    cam_rgba   = np.dstack((cm_rgb, alpha_mask))

    cam_pil = Image.fromarray(cam_rgba, "RGBA")
    cam_pil = cam_pil.resize((216, 216), Image.NEAREST)          # keep blocky
    size    = max(img.size)
    cam_pil = cam_pil.resize((size, size), Image.BICUBIC)
    cam_pil = transforms.CenterCrop(img.size[::-1])(cam_pil)     # exactly img.size
    return cam_pil

def _extract_overlay_pixels(composite: Image.Image, background: Image.Image) -> np.ndarray:
    """
    Given the CAM-composited RGBA image returned by `grad_cam()` and the
    original RGBA background, return a float32 array of shape (H, W, 4)
    that contains *only* the coloured heat-map (premultiplied RGB + α).

    Pixels where the composite == background have α = 0.
    """
    fg = np.asarray(composite.convert("RGBA"), dtype=np.float32)
    bg = np.asarray(background.convert("RGBA"), dtype=np.float32)

    # anything identical to the background means "no heat" → α = 0
    diff = np.any(fg != bg, axis=-1, keepdims=True).astype(np.float32)
    a    = diff[..., 0:1]            # 1 where heat present, else 0
    rgb  = fg[..., :3] - bg[..., :3] # colour relative to background
    rgb  = np.clip(rgb, 0, 255)      # safety clamp

    premul = np.concatenate([rgb * (a / 255.0), a], axis=-1) / 255.0
    return premul                    # H,W,4 float32 premult


def grad_cam(img: Image.Image, tag: str, m: torch.nn.Module,
             alpha: float, thr: float = 0.2) -> Image.Image:
    """Fully matches the HuggingFace demo Grad‑CAM pipeline."""
    dev = next(m.parameters()).device
    idx = TAGS[tag]

    tensor = TRANSFORM(img.convert("RGBA")).unsqueeze(0).to(dev)

    gradients, activations = {}, {}

    def hook_fwd(_, __, out):
        activations["value"] = out

    def hook_bwd(_, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = m.norm.register_forward_hook(hook_fwd)
    h2 = m.norm.register_full_backward_hook(hook_bwd)

    logits = m(tensor)[0]
    m.zero_grad()
    logits[idx].backward(retain_graph=True)

    with torch.no_grad():
        patch_grads = gradients["value"]          # (B, P, E)
        patch_acts  = activations["value"]        # (B, P, E)

        weights = patch_grads.mean(dim=1).squeeze(0)      # (E,)
        cam_1d = torch.einsum("pe,e->p", patch_acts.squeeze(0), weights)
        cam_1d = torch.relu(cam_1d)

        P = cam_1d.numel()
        h, w_ = _best_grid(P)                       # 27×27 for ViT‑384, 24×48 for SigLIP, etc.
        cam = cam_1d.reshape(h, w_).cpu().numpy()

    h1.remove(); h2.remove()

    overlay = create_cam_visualization_pil(img, cam, alpha=alpha, vis_threshold=thr)
    comp    = Image.alpha_composite(img.convert("RGBA"), overlay)
    return comp, overlay             # ← second result is the pure heat-map RGBA
# ╰─────────────────────────────────────╯


def grad_cam_average(img: Image.Image, tag: str, model_keys, alpha, thr=0.2):
    """
    1. Call grad_cam() for every selected model, but keep only the RGBA overlay.
    2. Convert each overlay to float premultiplied-RGBA.
    3. Average pixel-wise.
    4. Lay one averaged overlay back onto the untouched source image.
    """
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    overlays = []

    for k in model_keys:
        if not REG[k].get("supports_cam", True):
            continue
        model = load_model(k, dev)
        _, ov = grad_cam(img, tag, model, alpha=alpha, thr=thr)  # we need only ov
        overlays.append(np.asarray(ov, dtype=np.float32) / 255.0)

    if not overlays:
        return img

    # premultiply RGB by α, average, then un-premultiply
    premuls = []
    for arr in overlays:
        rgb, a = arr[..., :3], arr[..., 3:4]
        premuls.append(np.concatenate([rgb * a, a], axis=-1))

    avg = np.mean(premuls, axis=0)
    rgb_pm, a = avg[..., :3], np.clip(avg[..., 3:4], 1e-8, 1.0)
    rgb = (rgb_pm / a) * 255.0
    rgba = np.concatenate([rgb, a * 255.0], axis=-1).astype("uint8")
    overlay_avg = Image.fromarray(rgba, mode="RGBA")

    return Image.alpha_composite(img.convert("RGBA"), overlay_avg)





# -----------------------------------------------------------
# safe_cam() – returns a 2-D numpy array for one model
# -----------------------------------------------------------
# 2️⃣ _safe_cam  ───────────────────────────────────────────────
def _safe_cam(img: Image.Image, tag: str, model, idx: int):
    dev = next(model.parameters()).device
    t = TRANSFORM(img.convert("RGBA")).unsqueeze(0).to(dev).requires_grad_(True)

    acts, grads = {}, {}
    def fwd(_, __, o): acts["v"] = o
    def bwd(_, gi, go): grads["v"] = go[0]

    h1 = model.norm.register_forward_hook(fwd)
    h2 = model.norm.register_full_backward_hook(bwd)

    model.zero_grad()
    model(t)[0, idx].backward()

    h1.remove(); h2.remove()

    a = acts["v"].squeeze(0)
    g = grads["v"].squeeze(0)
    w = g.mean(1)
    cam_1d = torch.relu((w[:, None] * a).sum(0)).detach().cpu().numpy()

    h, w_ = _patch_grid_from_token_count(cam_1d.size, 384, 384)
    print(
        f"{model.__class__.__name__}:  "
        f"img_size={getattr(model.patch_embed, 'img_size', '?')}, "
        f"stride={getattr(model.patch_embed.proj, 'stride', '?')}, "
        f"P={cam_1d.size}, grid=({h},{w_})"
    )

    return cam_1d.reshape(h, w_)


# -----------------------------------------------------------
# grad_cam_multi() – average over all selected models
# -----------------------------------------------------------
def grad_cam_multi(event: gr.SelectData, img, model_keys, alpha, thr):
    tag_str = event.value
    print("clicked tag →", tag_str)

    dev  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cams = []

    for k in model_keys:
        if not REG[k].get("supports_cam", True):
            continue
        m   = load_model(k, dev)
        idx = TAGS.get(tag_str)
        if idx is None:
            continue
        cams.append(_safe_cam(img, tag_str, m, idx))

    if not cams:
        return img, {}

    cam  = np.mean(cams, axis=0)
    cam  = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cm_rgb     = (cm.inferno(cam)[..., :3] * 255).astype("uint8")
    alpha_mask = ((cam >= thr) * alpha * 255).astype("uint8")
    cam_rgba   = np.dstack((cm_rgb, alpha_mask)).copy()

    overlay = (Image.fromarray(cam_rgba, "RGBA")
               .resize((216, 216), Image.NEAREST)
               .resize((max(img.size),) * 2, Image.BICUBIC))
    overlay = transforms.CenterCrop(img.size[::-1])(overlay)
    return Image.alpha_composite(img.convert("RGBA"), overlay), {"tag": tag_str}

# ╭──────────── Batch worker ─────────────╮
def worker_loop(dev_key, q, model_keys, thr, out_root,
                total, counter, lock, progress):
    dev    = torch.device(dev_key)
    models = {k: load_model(k, dev) for k in model_keys}
    while True:
        try:
            p = q.get_nowait()
        except queue.Empty:
            return
        try:
            img = Image.open(p).convert("RGBA")
            t   = TRANSFORM(img).unsqueeze(0).to(dev)
            for k, m in models.items():
                tags, _ = classify_tensor(t, m, thr, REG[k]["head_type"], REG[k].get("backend", "pytorch"))
                out_dir = out_root / k
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{p.stem}.txt").write_text(tags, encoding="utf-8")
        except Exception as e:
            print(f"[WARN] {p.name} skipped: {e}")
        finally:
            with lock:
                counter[0] += 1
                progress((counter[0], total))
            q.task_done()
# ╰──────────────────────────────────────╯

# ╭──────────── Batch front - generator ─────────────╮
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

def batch_tag(folder, thr, model_keys, devices, cpu_cores,
              progress=gr.Progress(track_tqdm=True)):
    if not folder:
        yield "❌ No folder provided."; return
    in_dir = Path(folder).expanduser()
    if not in_dir.is_dir():
        yield f"❌ Not a directory: {in_dir}"; return

    imgs = [p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not imgs:
        yield f"⚠️ No images found in {in_dir}"; return

    out_root = in_dir / "tags"  # root container for all models
    out_root.mkdir(exist_ok=True)

    total   = len(imgs)
    counter = [0]
    lock    = threading.Lock()
    q = queue.Queue()  # standard FIFO
    for p in imgs:  # preload every path
        q.put(p)

    threads = []

    if "CPU" in devices and cpu_cores > 0:
        for _ in range(min(cpu_cores, os.cpu_count() or 1)):
            t = threading.Thread(target=worker_loop,
                    args=("cpu", q, model_keys, thr,
                          out_root, total, counter, lock, progress))
            t.start(); threads.append(t)

    for idx, lbl in enumerate(GPU_LABELS):
        if lbl in devices:
            t = threading.Thread(target=worker_loop,
                    args=(f"cuda:{idx}", q, model_keys, thr,
                          out_root, total, counter, lock, progress))
            t.start(); threads.append(t)

    for t in threads:
        t.join()

    yield f"✅ {total} images processed for models {', '.join(model_keys)}."
# ╰──────────────────────────────────────────────╯


def _pick_device(devices: list[str]) -> torch.device:
    for idx, lbl in enumerate(GPU_LABELS):
        if lbl in devices:
            return torch.device(f"cuda:{idx}")
    return torch.device("cpu")


def batch_caption(folder, caption_type, caption_length, extra_opts, name_field,
                  temperature, top_p, max_new_tokens, devices,
                  progress=gr.Progress(track_tqdm=True)):
    if not folder:
        yield "❌ No folder provided."; return
    in_dir = Path(folder).expanduser()
    if not in_dir.is_dir():
        yield f"❌ Not a directory: {in_dir}"; return

    imgs = [p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    if not imgs:
        yield f"⚠️ No images found in {in_dir}"; return

    out_dir = in_dir / "captions"
    out_dir.mkdir(exist_ok=True)

    device = _pick_device(devices)
    total = len(imgs)
    start = time.time()

    for i, p in enumerate(imgs, 1):
        img = Image.open(p).convert("RGB")
        prompt = build_prompt(caption_type, caption_length, extra_opts, name_field)
        caption = caption_once(img, prompt, temperature, top_p, max_new_tokens, device)
        (out_dir / f"{p.stem}.txt").write_text(caption, encoding="utf-8")
        eta = (time.time() - start) / i * (total - i)
        yield f"{i}/{total} done – ETA {int(eta)//60:02d}:{int(eta)%60:02d}"

    yield f"✅ Finished {total} images → {out_dir}"

CSS = """
.inferno-slider input[type=range]{background:linear-gradient(to right,#000004,#1b0c41,#4a0c6b,#781c6d,#a52c60,#cf4446,#ed6925,#fb9b06,#f7d13d,#fcffa4)!important}
#image_container-image{width:100%;aspect-ratio:1/1;max-height:100%}
#image_container img{object-fit:contain!important}
"""
demo = gr.Blocks(css=CSS)

with demo:
    gr.Markdown("## Data Curation Tool - Model Builder UI")

    orig_state  = gr.State()        # keeps the uploaded PIL image
    cam_state   = gr.State()        # keeps last Grad-CAM info
    scores_state= gr.State()        # NEW ─ keeps full {tag: score} dict

    model_menu = gr.Dropdown(
        choices=list(REG.keys()),
        value="pilot2",
        multiselect=True,
        label="Models to run"
    )

    # ─── Single-image tab
    with gr.Tab("Single Image"):
        with gr.Row():
            with gr.Column():
                with gr.Tab("Original"):
                    img_orig = gr.Image(         # new — shows uploaded image untouched
                        sources=["upload", "clipboard"],
                        type="pil",
                        label="Source",
                        elem_id="image_container",
                    )
                with gr.Tab("Grad-CAM"):
                    img_cam = gr.Image(          # new — shows heat-map overlay
                        type="pil",
                        label="CAM",
                        elem_id="image_container",
                    )
                cam_thr  = gr.Slider(0, 1, 0.4, 0.01, label="CAM threshold",
                                     elem_classes="inferno-slider")#.4
                cam_alpha= gr.Slider(0, 1, 0.6, 0.01, label="CAM alpha")#.6
            with gr.Column():
                thr_slider= gr.Slider(0, 1, 0.2, 0.01, label="Tag threshold")
                tag_out   = gr.Textbox(label="Tag string")
                interrogate_btn = gr.Button("Interrogate", variant="secondary")
                save_btn = gr.Button("Save current tags")
                download = gr.File(label="Download .txt", visible=False)
                lbl_out   = gr.Label(num_top_classes=250, show_label=False)

    # ─── Batch tagging tab
    with gr.Tab("Batch Folder"):
        folder_box = gr.Textbox(label="Folder path")
        thr_batch  = gr.Slider(0, 1, 0.2, 0.01, label="Tag threshold")
        dev_check  = gr.CheckboxGroup(["CPU"] + GPU_LABELS, value=["CPU"],
                                      label="Compute devices")
        cpu_slide  = gr.Slider(1, max(os.cpu_count() or 1, 1), 1, 1,
                               label="CPU cores", visible=True)
        run_btn    = gr.Button("Start batch tagging", variant="primary")
        status_box = gr.Textbox(label="Status", interactive=False)

    # ─── Captioner tab
    with gr.Tab("Captioner"):
        with gr.Tab("Single"):
            cap_image = gr.Image(type="pil", label="Input Image")
            cap_type = gr.Dropdown(choices=list(CAPTION_TYPE_MAP.keys()), value="Descriptive", label="Caption Type")
            cap_len = gr.Dropdown(choices=["any", "very short", "short", "medium-length", "long", "very long"] + [str(i) for i in range(20, 261, 10)], value="long", label="Caption Length")
            with gr.Accordion("Extra Options", open=False):
                cap_opts = gr.CheckboxGroup(choices=[NAME_OPTION], label="Select one or more")
            name_box = gr.Textbox(label="Person / Character Name", visible=False)
            cap_opts.change(toggle_name_box, cap_opts, name_box)
            with gr.Accordion("Generation settings", open=False):
                temp_slider = gr.Slider(0.0, 2.0, 0.6, 0.05, label="Temperature")
                top_p_slider = gr.Slider(0.0, 1.0, 0.9, 0.01, label="Top-p")
                max_tok_slider = gr.Slider(1, 2048, 512, 1, label="Max New Tokens")
            cap_devices = gr.CheckboxGroup(["CPU"] + GPU_LABELS, value=["CPU"], label="Compute devices")
            cap_btn = gr.Button("Caption")
            cap_out = gr.Textbox(label="Caption")
            cap_btn.click(
                caption_single,
                inputs=[cap_image, cap_type, cap_len, cap_opts, name_box, temp_slider, top_p_slider, max_tok_slider, cap_devices],
                outputs=cap_out,
            )

        with gr.Tab("Batch"):
            cap_folder = gr.Textbox(label="Input folder")
            cap_batch_btn = gr.Button("Run batch caption")
            cap_progress = gr.Textbox(label="Progress", interactive=False)
            cap_devices_b = gr.CheckboxGroup(["CPU"] + GPU_LABELS, value=["CPU"], label="Compute devices")
            cap_batch_btn.click(
                batch_caption,
                inputs=[cap_folder, cap_type, cap_len, cap_opts, name_box, temp_slider, top_p_slider, max_tok_slider, cap_devices_b],
                outputs=cap_progress,
            )

    # ─── OpenRouter API tab -------------------------------------------------
    add_openrouter_tab()
    # ─── Tag Cleaning Utility tab ------------------------------------------
    add_clean_tags_tabs()

    def save_tags(tag_string):
        if not tag_string:
            return gr.update(visible=False)
        fn = Path("tags_current.txt")
        fn.write_text(tag_string, encoding="utf-8")
        return gr.update(value=str(fn), visible=True)


    save_btn.click(save_tags, inputs=tag_out, outputs=download)

    # ── dynamic CPU-core slider
    dev_check.change(lambda sel: gr.update(visible=("CPU" in sel)),
                     dev_check, cpu_slide)

    # ─── Single-image aggregation over selected models ──────────────────────
    def single(img, thr, model_keys):
        if img is None:
            return "", {}, {}, img

        full_scores = defaultdict(list)

        for k in model_keys:
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            m   = load_model(k, dev)
            t   = TRANSFORM(img.convert("RGBA")).unsqueeze(0).to(dev)
            _, sc = classify_tensor(t, m, 0.0, REG[k]["head_type"], REG[k].get("backend", "pytorch"))      # keep all scores
            for tag, val in sc.items():
                full_scores[tag].append(val)

        agg = {tag: float(np.mean(vs)) for tag, vs in full_scores.items()}
        visible = {k: v for k, v in agg.items() if v > thr}
        tag_str = ", ".join(visible)

        return tag_str, visible, agg, img

    img_orig.upload(
        single,
        inputs=[img_orig, thr_slider, model_menu],
        outputs=[tag_out, lbl_out, scores_state, orig_state],
    )

    # update label when threshold slider moves
    thr_slider.input(
        lambda thr, scores: (
            ", ".join(k for k, v in scores.items() if v > thr),
            {k: v for k, v in scores.items() if v > thr}
        ) if scores else ("", {}),
        inputs=[thr_slider, scores_state],
        outputs=[tag_out, lbl_out],
    )

    def filter_by_threshold(thr, scores):
        if not scores:
            return "", {}
        visible = {k: v for k, v in scores.items() if v > thr}
        return ", ".join(visible), visible

    # ─── robust helper: works on every Gradio build ─────────────────────
    def _selected_tag(data: gr.SelectData):
        """
        Extract the tag string from whatever gr.Label.select() sends back.
        """
        # legacy object
        if hasattr(data, "value"):
            print(f"data.value:\t{data.value}")
            return data.value  # type: ignore[attr-defined]

        # new dict styles
        if isinstance(data, dict):
            for key in ("value", "label", "name", "item"):
                if key in data and isinstance(data[key], str):
                    return data[key]

            # score-dict: pick highest-scoring key
            try:
                return max(data.items(), key=lambda kv: kv[1])[0]
            except Exception:
                return next(iter(data))  # fallback: first key

        # already a string?
        if isinstance(data, str):
            return data

        return ""


    def interrogate(img, thr, model_keys):
        # Re-run inference using cached models
        return single(img, thr, model_keys)[:3]  # tag_out, lbl_out, scores_state


    interrogate_btn.click(
        interrogate,
        inputs=[orig_state, thr_slider, model_menu],
        outputs=[tag_out, lbl_out, scores_state],
    )


    def handle_cam_click(event: gr.SelectData,
                         alpha: float,
                         thr: float,
                         img: Image.Image,
                         model_keys: list[str]):
        tag = event.value
        print(f"tag:\t{tag}")
        if img is None or tag not in TAGS:
            return img, {}

        if len(model_keys) == 1:
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            m = load_model(model_keys[0], dev)
            comp, ov = grad_cam(img, tag, m, alpha, thr)
        else:
            comp = grad_cam_average(img, tag, model_keys, alpha, thr)

        return comp, {"tag": tag}

    def handle_cam_slider(
            tag,
            alpha: float,
            thr: float,
            img: Image.Image,
            model_keys: list[str],
    ):
        """
        Dispatches to either single-model Grad-CAM (grad_cam) or
        multi-model averaging (grad_cam_average), depending on how many
        models are selected in the dropdown.
        """
        tag = tag["tag"]
        print(f"tag:\t{tag}")
        if img is None or tag not in TAGS:
            return img, {}

        if len(model_keys) == 1:
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            m = load_model(model_keys[0], dev)
            comp, ov = grad_cam(img, tag, m, alpha, thr)
        else:
            comp = grad_cam_average(img, tag, model_keys, alpha, thr)

        return comp, {"tag": tag}

    lbl_out.select(
        fn=handle_cam_click,
        inputs=[cam_alpha, cam_thr, orig_state, model_menu],  # event comes first automatically
        outputs=[img_cam, cam_state],
    )

    # ─── move CAM-threshold slider ----------------------------------------------
    cam_thr.input(
        fn=handle_cam_slider,
        inputs=[cam_state, cam_alpha, cam_thr, orig_state, model_menu],  # event comes first automatically
        outputs=[img_cam, cam_state],
    )

    # ─── move CAM-alpha slider ---------------------------------------------------
    cam_alpha.input(
        fn=handle_cam_slider,
        inputs=[cam_state, cam_alpha, cam_thr, orig_state, model_menu],  # event comes first automatically
        outputs=[img_cam, cam_state],
    )

    # ───────────────── Batch callbacks ──────────────────────────────
    run_btn.click(
        batch_tag,
        inputs=[folder_box, thr_batch, model_menu, dev_check, cpu_slide],
        outputs=status_box,
    )

################################################################################
#                                      Main                                    #
################################################################################

if __name__ == "__main__":
    print("PyTorch", torch.__version__, "| CUDA", torch.version.cuda, "| GPUs", torch.cuda.device_count())
    gr.close_all()
    demo.launch(server_name="0.0.0.0", share=False, show_error=True)
