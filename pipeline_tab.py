import json
import shutil
from pathlib import Path
from typing import List, Tuple, Dict

import gradio as gr

from user_config import load_config, update_config
from openrouter_tab import process_batch, IMG_EXTS
from tag_cleaner_app import merge_tag_files

__all__ = ["add_pipeline_tab"]

STEP_FOLDERS = [
    "01_raw_input",
    "02_captioned",
    "03_cleaned",
    "04_pruned",
    "05_review",
    "06_final",
]


def _ensure_stage_dirs(root: Path) -> List[Path]:
    dirs = []
    for name in STEP_FOLDERS:
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)
    return dirs


def _prepend_trigger(folder: Path, trigger: str) -> None:
    for txt in folder.glob("*.txt"):
        content = txt.read_text(encoding="utf-8")
        if not content.startswith(trigger):
            txt.write_text(f"{trigger}, {content}", encoding="utf-8")


def _ensure_images(tag_dir: Path, img_src: Path) -> None:
    """Ensure every tag file in tag_dir has its matching image.

    Copies images from img_src when absent so OpenRouter can process the folder
    with image+tag pairs.
    """
    tag_dir.mkdir(parents=True, exist_ok=True)
    for txt in tag_dir.glob("*.txt"):
        stem = txt.stem
        have_img = any((tag_dir / f"{stem}{ext}").exists() for ext in IMG_EXTS)
        if have_img:
            continue
        for ext in IMG_EXTS:
            src = img_src / f"{stem}{ext}"
            if src.exists():
                shutil.copy(src, tag_dir / src.name)
                break



def ingest_scrape(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for img in src.glob("*"):
        if img.suffix.lower() in IMG_EXTS:
            shutil.copy(img, dest / img.name)
            meta_json = img.with_suffix(".json")
            meta_txt = img.with_suffix(".txt")
            tag_file = dest / f"{img.stem}.txt"
            if meta_json.exists():
                try:
                    data = json.loads(meta_json.read_text())
                    tags = data.get("tags") or data.get("keywords") or []
                    if isinstance(tags, str):
                        tags = tags.split(",")
                    tag_file.write_text(", ".join(t.strip() for t in tags if t.strip()), encoding="utf-8")
                except Exception as e:
                    gr.logger.warning("Failed to parse %s: %s", meta_json, e)
            elif meta_txt.exists():
                tag_file.write_text(meta_txt.read_text(encoding="utf-8"), encoding="utf-8")
def _stage_caption(api_key: str, model: str, prompt: str, folder: Path) -> str:
    return process_batch(api_key, "Generate Tags from Image (Image \u2794 Tags)", prompt, str(folder), model)


def _stage_prune(api_key: str, model: str, prompt: str, folder: Path) -> str:
    return process_batch(api_key, "Filter Existing Tags (Image + Tags \u2794 Pruned Tags)", prompt, str(folder), model)


def _parse_models(text: str) -> List[dict]:
    """Return list of model configs from JSON or comma-separated text."""
    if not text.strip():
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict) and d.get("model")]
    except Exception:
        pass
    models = [m.strip() for m in text.replace("\n", ",").split(",") if m.strip()]
    return [{"model": m} for m in models]


def _copy_all(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.txt"):
        shutil.copy(f, dst / f.name)


def _caption_multiple(api_key: str, cfgs: List[dict], raw_dir: Path, out_dir: Path, prompt_fallback: str) -> Tuple[List[Path], list[str]]:
    folders = []
    msgs = []
    for cfg in cfgs:
        model = cfg.get("model")
        prompt = cfg.get("prompt", prompt_fallback)
        if not model:
            continue
        res = _stage_caption(api_key, model, prompt or "", raw_dir)
        msgs.append(f"[{model}] {res}")
        gen_dir = raw_dir / "generated_tags"
        dest = out_dir / Path(model).name
        if gen_dir.exists():
            _copy_all(gen_dir, dest)
            shutil.rmtree(gen_dir)
        folders.append(dest)
    return folders, msgs


def _prune_multiple(api_key: str, cfgs: List[dict], in_dir: Path, out_dir: Path, prompt_fallback: str) -> Tuple[List[Path], list[str]]:
    folders = []
    msgs = []
    for cfg in cfgs:
        model = cfg.get("model")
        prompt = cfg.get("prompt", prompt_fallback)
        if not model:
            continue
        res = _stage_prune(api_key, model, prompt or "", in_dir)
        msgs.append(f"[{model}] {res}")
        filt_dir = in_dir / "filtered_tags"
        dest = out_dir / Path(model).name
        if filt_dir.exists():
            _copy_all(filt_dir, dest)
            shutil.rmtree(filt_dir)
        folders.append(dest)
    return folders, msgs


def _collect_review_items(root: Path) -> List[Tuple[Path, List[str], Dict[str, List[str]]]]:
    img_dir = root / STEP_FOLDERS[0]
    pruned_dir = root / STEP_FOLDERS[3]
    model_dirs = [d for d in pruned_dir.iterdir() if d.is_dir()]
    items = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        model_tags: Dict[str, List[str]] = {}
        union: set[str] = set()
        for d in model_dirs:
            txt = d / f"{img.stem}.txt"
            if txt.exists():
                tags = [t.strip() for t in txt.read_text(encoding='utf-8').split(',') if t.strip()]
                model_tags[d.name] = tags
                union.update(tags)
        if not model_dirs:
            txt = root / STEP_FOLDERS[4] / 'MERGED_TAG_RESULTS' / f"{img.stem}.txt"
            if txt.exists():
                union.update([t.strip() for t in txt.read_text(encoding='utf-8').split(',') if t.strip()])
        items.append((img, sorted(union), model_tags))
    return items


def run_pipeline(
    root: str,
    scrape_src: str,
    caption_cfg: str,
    caption_prompt: str,
    prune_cfg: str,
    prune_prompt: str,
    trigger: str,
    api_key: str,
) -> str:
    root_path = Path(root).expanduser()
    update_config(
        "pipeline",
        root=root,
        scrape=scrape_src,
        caption_model=caption_cfg,
        prune_model=prune_cfg,
        trigger=trigger,
    )
    s1, s2, s3, s4, s5, s6 = _ensure_stage_dirs(root_path)
    if scrape_src:
        ingest_scrape(Path(scrape_src).expanduser(), s1)

    msg: list[str] = []
    msg.append(f"Folders prepared under {root_path}.")

    caption_models = _parse_models(caption_cfg)
    prune_models = _parse_models(prune_cfg)

    caption_dirs: List[Path] = []
    if api_key and caption_models:
        caption_dirs, notes = _caption_multiple(api_key, caption_models, s1, s2, caption_prompt)
        msg.extend(notes)
    else:
        msg.append("Skipped captioning (missing API key or models).")

    merged_msg, merged_path_str = merge_tag_files(caption_dirs or [s1], s3)
    msg.append(merged_msg)
    merged_path = Path(merged_path_str)
    _ensure_images(merged_path, s1)

    prune_dirs: List[Path] = []
    if api_key and prune_models:
        prune_dirs, p_notes = _prune_multiple(api_key, prune_models, merged_path, s4, prune_prompt)
        msg.extend(p_notes)
    else:
        msg.append("Skipped pruning (missing API key or models).")

    merge_p_msg, review_path_str = merge_tag_files(prune_dirs or [merged_path], s5)
    msg.append(merge_p_msg)
    review_path = Path(review_path_str)
    _prepend_trigger(review_path, trigger)
    msg.append(f"Review folder ready at {review_path}")
    return "\n".join(msg)


def add_pipeline_tab():
    cfg = load_config().get("pipeline", {})
    with gr.Tab("Automation Pipeline"):
        root_in = gr.Textbox(label="Dataset Root Folder", value=cfg.get("root", ""))
        scrape_in = gr.Textbox(label='Scraped Data Folder', value=cfg.get('scrape', ''))
        caption_model_in = gr.Textbox(label="Caption Model Config", value=cfg.get("caption_model", ""))
        caption_prompt_in = gr.Textbox(label="Caption Prompt", lines=2)
        prune_model_in = gr.Textbox(label="Prune Model Config", value=cfg.get("prune_model", ""))
        prune_prompt_in = gr.Textbox(label="Prune Prompt", lines=2)
        trigger_in = gr.Textbox(label="LoRA Trigger", value=cfg.get("trigger", ""))
        api_key_in = gr.Textbox(label="OpenRouter API Key", type="password")
        run_btn = gr.Button("Run Pipeline", variant="primary")
        out_box = gr.Textbox(label="Status", lines=10)

        review_btn = gr.Button("Load Review Items")
        idx_slider = gr.Slider(label="Image Index", minimum=0, maximum=0, step=1, value=0)
        gallery = gr.Gallery(label="Images", columns=4, height="auto")
        img_out = gr.Image()
        tags_chk = gr.CheckboxGroup(label="Tags to Keep")
        save_btn = gr.Button("Save Current Tags")
        finalize_btn = gr.Button("Finalize Dataset")
        review_status = gr.Textbox(label="Review Status")
        model_box = gr.Textbox(label='Tags by Model', lines=4)
        data_state = gr.State([])

        run_btn.click(
            run_pipeline,
            inputs=[root_in, scrape_in, caption_model_in, caption_prompt_in,
                    prune_model_in, prune_prompt_in, trigger_in, api_key_in],
            outputs=out_box,
        )

        def _load(root):
            items = _collect_review_items(Path(root))
            if not items:
                return (
                    gr.update(minimum=0, maximum=0, value=0),
                    None,
                    gr.update(choices=[], value=[]),
                    "",
                    "No items found.",
                    [],
                    []
                )
            img, tags, by_model = items[0]
            gallery_imgs = [i for i, *_ in items]
            return (
                gr.update(minimum=0, maximum=len(items)-1, value=0),
                img,
                gr.update(choices=tags, value=tags),
                json.dumps({k: ', '.join(v) for k, v in by_model.items()}, indent=2),
                "Loaded",
                items,
                gallery_imgs,
            )

        review_btn.click(
            _load,
            inputs=[root_in],
            outputs=[idx_slider, img_out, tags_chk, model_box, review_status, data_state, gallery],
        )

        def _update(idx, data):
            if not data:
                return None, gr.update(choices=[], value=[]), ''
            img, tags, by_model = data[int(idx)]
            return img, gr.update(choices=tags, value=tags), json.dumps({k: ', '.join(v) for k, v in by_model.items()}, indent=2)

        idx_slider.change(_update, inputs=[idx_slider, data_state], outputs=[img_out, tags_chk, model_box])

        def _select(evt: gr.SelectData, data):
            if not data:
                return gr.update(value=0), None, gr.update(choices=[], value=[]), ''
            idx = evt.index
            img, tags, by_model = data[idx]
            return (
                gr.update(value=idx),
                img,
                gr.update(choices=tags, value=tags),
                json.dumps({k: ', '.join(v) for k, v in by_model.items()}, indent=2),
            )

        gallery.select(_select, inputs=[data_state], outputs=[idx_slider, img_out, tags_chk, model_box])

        def _save(idx, tags, data, root):
            if not data:
                return 'Nothing loaded', data
            idx = int(idx)
            img, _, by_model = data[idx]
            review_dir = Path(root) / STEP_FOLDERS[4] / 'MERGED_TAG_RESULTS'
            review_dir.mkdir(parents=True, exist_ok=True)
            (review_dir / f'{img.stem}.txt').write_text(', '.join(tags), encoding='utf-8')
            data[idx] = (img, list(tags), by_model)
            return 'Saved', data

        save_btn.click(
            _save,
            inputs=[idx_slider, tags_chk, data_state, root_in],
            outputs=[review_status, data_state],
        )

        def _finalize(root, trigger):
            review_dir = Path(root) / STEP_FOLDERS[4] / "MERGED_TAG_RESULTS"
            final_dir = Path(root) / STEP_FOLDERS[5]
            final_dir.mkdir(parents=True, exist_ok=True)
            _prepend_trigger(review_dir, trigger)
            _copy_all(review_dir, final_dir)
            src_imgs = Path(root) / STEP_FOLDERS[0]
            for img in src_imgs.iterdir():
                if img.suffix.lower() in IMG_EXTS:
                    shutil.copy(img, final_dir / img.name)
            count = len(list(review_dir.glob('*.txt')))
            return f"Finalized {count} files."

        finalize_btn.click(_finalize, inputs=[root_in, trigger_in], outputs=review_status)

