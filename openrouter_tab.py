import base64
import io
import json
import time
from pathlib import Path
from typing import List, Tuple

import gradio as gr
import requests
from PIL import Image
from tqdm import tqdm

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-flash-1.5"
MODEL_CHOICES = [
    "openai/gpt-4o",
    "google/gemini-flash-1.5",
    "google/gemini-flash-1.5-8b",
    "google/gemini-pro-1.5",
]
SUBFOLDER_FILTERED = "filtered_tags"
SUBFOLDER_GENERATED = "generated_tags"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

SYSTEM_PROMPT_BASE = """
Your task is to describe every aspect, object, and interaction within this image, such that a blind person could perfectly capture it within their imagination if read aloud. You need to do it multiple times, each one a different \"style\" of description. No intro/outro, just give several styles.
    \"Follow Character-LoRA tagging rules: \"
    \"\u2022 no prefixes/suffixes other than character-name, series, species, etc. \"
    \"\u2022 no style adjectives (e.g. 'high quality', 'masterpiece') \"
    \"\u2022 lowercase, single words where possible \"
    \"\u2022 use underscores_not_spaces \"
    \"\u2022 (((return ONLY a comma-separated list of tags, nothing else.)))""".strip()

CONFIG_PATH = Path(__file__).with_name("session_config.json")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def save_config(api_key: str, model: str, mode: str, prompt_extra: str):
    data = {
        "api_key": api_key.strip(),
        "model": model,
        "mode": mode,
        "prompt_extra": prompt_extra,
    }
    CONFIG_PATH.write_text(json.dumps(data, indent=2))


def read_file_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip(), encoding="utf-8")


def img_to_b64(img_path: Path, max_side: int = 1024, quality: int = 85) -> str:
    img = Image.open(img_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    scale = max(img.size) / max_side
    if scale > 1:
        new_size = (int(img.width / scale), int(img.height / scale))
        img = img.resize(new_size, Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def data_url_from_b64(b64: str) -> str:
    return f"data:image/png;base64,{b64}"


def call_openrouter(api_key: str, messages: list, model: str,
                    temperature: float = 0.4, retries: int = 2) -> str:
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    for attempt in range(retries + 1):
        r = requests.post(ENDPOINT, headers=headers, json=body, timeout=180)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code == 503 and attempt < retries:
            time.sleep(2 + 2 * attempt)
            continue
        raise RuntimeError(f"OpenRouter {r.status_code}: {r.text}")


def normalize_tag_string(s: str) -> str:
    parts = [t.strip() for t in s.split(',') if t.strip()]
    return ', '.join(parts)


def fix_delimiters_in_folder(folder: str) -> str:
    root = Path(folder).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    changed, unchanged = 0, 0
    for txt in root.rglob('*.txt'):
        original = txt.read_text(encoding='utf-8', errors='ignore')
        normalized = normalize_tag_string(original)
        if normalized != original:
            txt.write_text(normalized, encoding='utf-8')
            changed += 1
        else:
            unchanged += 1
    return f"\u2713 {changed} files updated, {unchanged} already OK in \u201c{root}\u201d (incl. sub-folders)."


def process_single(api_key: str, mode: str, prompt_extra: str, image_file, tag_file, model: str) -> Tuple[str, str]:
    if not api_key:
        raise ValueError("API key is missing.")
    if mode.startswith("Filter") and tag_file is None:
        raise ValueError("In Filter mode you must provide a tag file.")
    img_path = Path(image_file)
    b64img = img_to_b64(img_path)
    system_text = SYSTEM_PROMPT_BASE
    user_text = prompt_extra.strip() + "\n\n" if prompt_extra.strip() else ""
    if mode.startswith("Filter"):
        tag_string = read_file_utf8(Path(tag_file))
        user_text += f"{normalize_tag_string(tag_string)}"
    else:
        user_text += "Look at the image and produce the most appropriate Character-LoRA tag string."
    messages = [
        {"role": "system", "content": system_text},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url_from_b64(b64img)}},
            ],
        },
    ]
    response = call_openrouter(api_key, messages, model=model)
    first_line = response.splitlines()[0]
    cleaned = first_line.strip().removeprefix('-').strip()
    return cleaned, response


def process_batch(api_key: str, mode: str, prompt_extra: str, folder: str, model: str) -> str:
    folder = Path(folder).expanduser()
    if not folder.exists():
        raise FileNotFoundError(folder)
    dest = folder / (SUBFOLDER_FILTERED if mode.startswith("Filter") else SUBFOLDER_GENERATED)
    processed = 0
    failures: List[str] = []
    images = [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS]
    for img in tqdm(images, desc="Processing"):
        txt = img.with_suffix('.txt')
        try:
            new_tags, _ = process_single(api_key, mode, prompt_extra, img, txt if txt.exists() else None, model)
            write_text_utf8(dest / txt.name, new_tags)
            processed += 1
        except Exception as e:
            failures.append(f"{img.name}: {e}")
    summary = f"\u2714 {processed} images processed.\n"
    if failures:
        summary += f"\u2718 {len(failures)} failed:\n" + "\n".join(failures[:20])
    return summary


def add_openrouter_tab():
    cfg = load_config()
    with gr.Tab("OpenRouter Tagger"):
        gr.Markdown("## Character-LoRA Tagging / Tag-Refinement Tool")
        with gr.Row():
            api_key_in = gr.Textbox(label="OpenRouter API Key", type="password", value=cfg.get("api_key", ""))
            model_drop = gr.Dropdown(MODEL_CHOICES, value=cfg.get("model", DEFAULT_MODEL), label="LLM Model")
            mode_radio = gr.Radio([
                "Filter Existing Tags (Image + Tags \u2794 Pruned Tags)",
                "Generate Tags from Image (Image \u2794 Tags)",
            ], label="Mode", value=cfg.get("mode", "Filter Existing Tags (Image + Tags \u2794 Pruned Tags)"))
        prompt_in = gr.Textbox(label="Extra instructions for the LLM (optional)", lines=3, value=cfg.get("prompt_extra", ""))
        gr.Markdown("### Single Image")
        with gr.Row():
            image_in = gr.Image(label="Image", type="filepath")
            tag_file_in = gr.File(label="Tag File (.txt)", file_types=["text"], interactive=True)
        run_single_btn = gr.Button("Run (Single Image)", variant="primary")
        single_out = gr.Textbox(label="New Tag String", show_copy_button=True)
        gr.Markdown("---\n### Batch Mode")
        batch_folder_in = gr.Textbox(label="Folder containing images + .txt tag files", placeholder=r"C:\\path\\to\\dataset")
        run_batch_btn = gr.Button("Run (Batch)", variant="primary")
        batch_out = gr.Textbox(label="Batch Summary", lines=8, interactive=False)
        gr.Markdown("### Post-process Tag Files")
        fix_btn = gr.Button("Fix Delimiters in Tag Files", variant="secondary")
        fix_out = gr.Textbox(label="Fix Summary", lines=3, interactive=False)

        def _run_fix(folder):
            return fix_delimiters_in_folder(folder)

        fix_btn.click(_run_fix, inputs=[batch_folder_in], outputs=[fix_out])

        def _run_single(api_key, model, mode, prompt_extra, img, txt):
            tags, _ = process_single(api_key, mode, prompt_extra or "", img, txt, model)
            save_config(api_key, model, mode, prompt_extra or "")
            return tags

        run_single_btn.click(_run_single, inputs=[api_key_in, model_drop, mode_radio, prompt_in, image_in, tag_file_in], outputs=[single_out])

        def _run_batch(api_key, model, mode, prompt_extra, folder):
            summary = process_batch(api_key, mode, prompt_extra or "", folder, model)
            save_config(api_key, model, mode, prompt_extra or "")
            return summary

        run_batch_btn.click(_run_batch, inputs=[api_key_in, model_drop, mode_radio, prompt_in, batch_folder_in], outputs=[batch_out])

        def toggle_tag_file(mode):
            need_tag = mode.startswith("Filter")
            return gr.update(interactive=need_tag, visible=need_tag)

        mode_radio.change(toggle_tag_file, inputs=[mode_radio], outputs=[tag_file_in])

