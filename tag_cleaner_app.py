"""
Tag-Cleaner GUI
─────────────────────────────────────────────────────────────
Buttons
  • Clean Files
  • Pad Whitespace Only
  • E621 Sanitize Tags
  • Merge Tag Files      ◀ NEW
"""

# ─── warm-up NuMind LLM so it’s ready when the user clicks ───
from artist_llm import MODELS, DEFAULT_ALIAS, extract_body, get_artists, _load_model
# _load_model()                                # prints a note & shows HF tqdm

from pathlib import Path
import re, itertools, collections
import gradio as gr
from e6_tag_utils import sanitize_tag_list

# existing std-lib imports …
import re, itertools, collections, json, torch       #  ← add torch
from pathlib import Path
from typing import List                              #  ← new

# third-party …
from PIL import Image                                #  ← new

# ─── add near the other imports ─────────────────────────────────
from typing import List
from PIL import Image
# ─── helper: single LLM call that returns comma-separated tags ──
PRUNE_RULES = """
You are a tag-pruning assistant for Stable Diffusion *character* LoRA datasets. Follow every rule below and reply with a single comma-separated line of tags—no other text.

RULES
You are a tag-pruning assistant for Stable Diffusion *character* LoRA datasets.  
The user will send **one line of comma-separated tags**. Perform the following steps and reply with a **single comma-separated line**—no other text.

1. Activation token first  
   • Identify and keep the unique activation tag that names the character; place it at position #1.

2. Prune constant character traits  
   • Remove tags that are always true for this character (hair/eye colour, species, default outfit, gender, body type, etc.).

3. Keep only desired variable context  
   • Preserve tags that vary across images (pose, outfit, prop, scenery, lighting, camera angle, rating, etc.) **only if** you want that variation retained.  
   • Drop any tag the dataset should learn as part of the activation token.

4. Danbooru normalisation  
   • Lower-case ASCII.  
   • Replace spaces with underscores.  
   • Prefer singular nouns.  
   • Remove duplicates.  
   • After the activation token, sort remaining tags alphabetically.

5. Length cap  
   • Final list must contain **3 – 15 tags total** (including the activation token).  
   • If over the limit, remove the least-important context tags until within the range.

6. Disallowed content  
   • Do **not** output URLs, hashes, brackets, quotes, or any text that is not a valid Danbooru-style tag.

7. Strict output format of tags (TO REMOVE)  
   <tag2>, <tag3>, … <tagN>  
   • Single line, comma + single-space delimiter.  
   • No leading/trailing spaces, no trailing comma, no punctuation or commentary before/after the list.

(IMPORTANT final rules):
    • NO echoing the rules allowed.
    • NO echoing the input text.
    • *ONLY* provide the output text.
    • *ONLY* PROVIDE THE TAGS THAT YOU THINK SHOULD BE REMOVED
"""







# ─── helper: find (image, text) pairs with the same basename ───
from typing import List, Tuple
from pathlib import Path

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

# ─── helper that builds the paired list ─────────────────────────
def _pair_files(img_dir: Path, txt_dir: Path) -> list[tuple[Path, Path]]:
    """Return [(txt_path, img_path), …] for every matching basename."""
    out = []
    for txt in txt_dir.glob("*.txt"):
        stem = txt.stem
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            img = img_dir / f"{stem}{ext}"
            if img.exists():
                out.append((txt, img))
                break
    return sorted(out, key=lambda t: t[0].name)   # stable order


# ------------------------------------------------------------------
#  robust llm_prune  – handles models w/ or w/o a "system" role
# ------------------------------------------------------------------
import re, torch, jinja2
from typing import List

_TAG_LINE_RX = re.compile(r"^[\w\- ]+,\s*[\w\- ]+")   # crude “tag, tag”

def _render(prompt_msgs, tok):
    """Try chat-template; if it fails, re-render w/o system role."""
    try:
        return tok.apply_chat_template(prompt_msgs, tokenize=False)
    except (ValueError, jinja2.exceptions.TemplateError):
        # concat rules → 1st user msg
        merged = {"role": "user",
                  "content": prompt_msgs[0]["content"] + "\n" +
                             prompt_msgs[1]["content"]}
        return tok.apply_chat_template([merged], tokenize=False)

import re, textwrap

# ─── llm_prune  (NEW version) ────────────────────────────────────
# -------------------------------------------------------------
# llm_prune  – model returns *TAGS TO REMOVE*                 |
# -------------------------------------------------------------
def llm_prune(caption_text: str,
              caption_tags: List[str],        # ← NEW
              tok,
              model,
              temp: float = 0.15,
              top_p: float = 1.0) -> List[str]:

    # --- build a plain prompt --------------------------------
    prompt = PRUNE_RULES + "\n" + caption_text.strip()

    with torch.inference_mode():
        out_ids = model.generate(
            **tok(prompt, return_tensors="pt").to(model.device),
            max_new_tokens=128,
            do_sample=False,
            temperature=temp,
            top_p=top_p,
        )

    # --- dump raw model reply for debugging ------------------
    raw_out = tok.decode(out_ids[0], skip_special_tokens=True)
    print("\n─── LLM  RAW  OUTPUT ───────────────────────────────")
    print(raw_out)
    print("────────────────────────────────────────────────────\n")

    # --- pull the fenced block after **Output:**  ------------
    extracted = ""
    m_head = re.search(r"\*\*Output:\*\*", raw_out, flags=re.I)
    if m_head:
        seg = raw_out[m_head.end():]
        try:
            extracted = seg.split("```", 2)[1].strip()
        except IndexError:
            pass

    # fallback – first ```code``` block
    if not extracted:
        m_block = re.search(r"```(?:\w*\n)?(.+?)```", raw_out, flags=re.S)
        if m_block:
            extracted = m_block.group(1).strip()

    # last fallback – anything after “removed”
    if not extracted:
        m_line = re.search(r"removed\s*(?:are|:)\s*(.+)", raw_out, flags=re.I | re.S)
        extracted = (m_line.group(1).strip() if m_line else raw_out).strip()

    # clean → list
    extracted = re.sub(r"^[\-\*\•\s]+", "", extracted)          # strip bullets
    cand_tags = [t.strip() for t in extracted.split(",") if t.strip()]

    # intersect with real caption tags (no hallucinations)
    real_set  = {t.lower() for t in caption_tags}
    to_remove = [t for t in cand_tags if t.lower() in real_set]

    print(f"🎯  Tags sent to checkbox ({len(to_remove)}): {to_remove}")
    print("========================================================\n")
    return to_remove



# ── utils.py (or put this near the top of app.py) ─────────────────
def safe_progress(prog, current, total=None, desc=None):
    if prog is None:
        return
    try:                                      # new Gradio (keywords ok)
        prog(current=current, total=total, desc=desc)
    except TypeError:                         # old Gradio
        # only value + optional description are allowed
        if desc is not None:
            prog(current, desc)
        else:
            prog(current)



# ────────────────── tiny helpers used throughout ──────────────────
def _split_tags(text: str) -> list[str]:
    return text.rstrip().split(", ")

def _dedup_order(tags: list[str]) -> list[str]:
    seen = set()
    return [t for t in tags if not (t in seen or seen.add(t))]

def _strip_eot(text: str) -> str:
    return re.sub(r"\s*<\|eot_id\|>\s*$", "", text.rstrip(), flags=re.S)

# original “clean”
def basic_clean(text: str) -> str:
    tags = _split_tags(_strip_eot(text))
    tags = [t.split(":")[-1].strip() if ":" in t else t for t in tags]
    return ", ".join(_dedup_order(tags))

# pad only
def pad_only(text: str) -> str:
    tags = [re.sub(r"\s+", "_", t) for t in _split_tags(_strip_eot(text))]
    return ", ".join(tags)

# ───────────────────── folder-level runners ───────────────────────
def run_over_folder(src: Path, sub: str, transform, prog: gr.Progress | None = None):
    if not src.is_dir():
        raise FileNotFoundError(f"'{src}' is not a directory")
    dst = src / sub
    dst.mkdir(exist_ok=True)

    files = [p for p in src.iterdir() if p.suffix.lower() == ".txt"]
    if not files:
        return "No .txt files found.", str(dst)

    for i, f in enumerate(files, 1):
        if prog:
            safe_progress(i, len(files), f"Processing {f.name}")
        try:
            raw = f.read_text(encoding="utf-8")
            new = transform(raw)
            (dst / f.name).write_text(new, encoding="utf-8")
        except Exception as e:
            gr.logger.warning("Skipped %s: %s", f.name, e)

    return f"✅  {len(files)} files processed ➜ {dst}", str(dst)

# ─────────────────────── merge feature ────────────────────────────
def merge_tag_files(in_dirs: list[Path], out_dir: Path, prog: gr.Progress | None = None):
    if not in_dirs:
        return "No input folders provided.", ""
    for d in in_dirs:
        if not d.is_dir():
            return f"'{d}' is not a folder.", ""

    target = out_dir / "MERGED_TAG_RESULTS"
    target.mkdir(parents=True, exist_ok=True)

    # gather set of all file names that exist in *any* folder
    all_names = sorted({p.name for d in in_dirs for p in d.glob("*.txt")})
    total = len(all_names)
    if total == 0:
        return "No .txt files found in any folder.", str(target)

    for idx, name in enumerate(all_names, 1):
        if prog:
            safe_progress(idx, total, f"Merging {name}")
        merged: list[str] = []
        for d in in_dirs:
            f = d / name
            if f.exists():
                tags = _split_tags(_strip_eot(f.read_text(encoding="utf-8")))
                merged.extend(tags)
        merged = _dedup_order(merged)
        (target / name).write_text(", ".join(merged), encoding="utf-8")

    return f"✅  Merged {total} files ➜ {target}", str(target)

# ─── JSON-to-caption injector ─────────────────────────────────────
import json, re
from pathlib import Path

_ARTIST_RE = lambda s: re.compile(rf"\b{re.escape(s)}\b", re.I)

def _inject_artist_year(txt_path: Path, artist: str, year: str) -> bool:
    """
    Insert  "<artist>, <year>, "  (both lower-cased) right after the first
    ", " in the file; prepend if none exists. Return True when modified.
    """
    artist_lc, year_lc = artist.lower().strip(), year.lower().strip()
    txt = txt_path.read_text(encoding="utf-8")

    # bail if BOTH already present (case-insensitive)
    if _ARTIST_RE(artist_lc).search(txt) and _ARTIST_RE(year_lc).search(txt):
        return False

    if ", " in txt:
        prefix, suffix = txt.split(", ", 1)
        new_txt = f"{prefix}, {artist_lc}, {year_lc}, {suffix}"
    else:
        new_txt = f"{artist_lc}, {year_lc}, {txt}"

    txt_path.write_text(new_txt, encoding="utf-8")
    return True


def inject_from_json(meta_json: Path, caption_dir: Path, prog=None) -> str:
    """Process the meta-json and patch caption *.txt files in-place."""
    data = json.load(meta_json.open(encoding="utf-8"))
    entries = [p for p in data.values() if isinstance(p, dict)]
    total   = len(entries)

    modified = missing = skipped = 0
    for i, post in enumerate(entries, 1):
        if prog: safe_progress(prog, i, total, f"Patching {post.get('filename','?')}.txt")

        artist   = str(post.get("artist",  "")).strip()
        year     = str(post.get("year",    "")).strip()
        filename = str(post.get("filename","")).strip()

        if not artist or not year or not filename:
            skipped += 1
            continue

        txt_path = caption_dir / f"{filename}.txt"
        if not txt_path.exists():
            missing += 1
            continue

        if _inject_artist_year(txt_path, artist, year):
            modified += 1

    return (f"✅ {modified} file(s) updated  │  "
            f"{missing} missing  │  {skipped} skipped (no data)")


YEAR_RX = re.compile(r"^\d{4}$")          # 4-digit year

def run_uploader_year(folder, out_dir, prog=gr.Progress()):
    folder = Path(folder).expanduser()
    files  = sorted(folder.glob("*.txt"))
    if not files:
        return "⚠️  No .txt files found", ""

    out_dir = Path(out_dir or folder / "UPLOADER_YEAR").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(files)
    for i, f in enumerate(files, 1):
        raw = f.read_text(encoding="utf-8")

        # ------------------------------------------------------------------
        # ① skip to first char *after* the first comma + any following “, ”s
        pos = raw.find(",")
        while pos + 1 < len(raw) and raw[pos+1] in {",", " ", "\t", "\n"}:
            pos += 1
        after = raw[pos+1:] if pos != -1 else raw                    # slice

        # ② split on the canonical delimiter
        parts = [p.strip() for p in after.split(", ") if p.strip()]
        if len(parts) < 3:
            safe_progress(prog, i, total, f"⚠️  Skipped {f.name}")
            continue

        # ③ locate the first 4-digit year and grab uploader = part before it
        year_idx = next((k for k, p in enumerate(parts) if YEAR_RX.match(p)), None)
        if year_idx is None or year_idx == 0:               # no upstream tag
            safe_progress(prog, i, total, f"⚠️  Skipped {f.name}")
            continue

        uploader = parts[year_idx - 1].lower().replace(" ", "_")
        year     = parts[year_idx]

        # ④ write "<uploader>, <year>" to new file
        (out_dir / f.name).write_text(f"{uploader}, {year}", encoding="utf-8")
        safe_progress(prog, i, total, f"Done {f.name}")

    return f"✅  Finished – {total} files", str(out_dir)

# ────────────────────────────────────────────────────────────────
#  Generic “restore_names”  – keeps original extension intact
# ────────────────────────────────────────────────────────────────
def restore_names(src_dir: Path,
                  out_sub: str                = "restored_names",
                  exts: list[str] | None      = None,
                  prog: gr.Progress | None    = None):
    """
    Copy files from *src_dir* into *src_dir/out_sub*, renaming them so that:

      1.  If the stem contains **"frame_"**, keep everything up to (but not
          including) the first '-' that *follows* that token.
      2.  Else, find the first '-' that *follows* an '=' sign and keep
          everything up to that dash.
      3.  If neither pattern matches, leave the stem unchanged.

    `exts` – optional list of extensions to process; if None, all files.
    """
    if not src_dir.is_dir():
        raise FileNotFoundError(f"'{src_dir}' is not a directory")

    # pick files
    files = [p for p in src_dir.iterdir()
             if p.is_file() and (exts is None or p.suffix.lower() in exts)]
    if not files:
        return "No matching files found.", str(src_dir)

    out_dir = src_dir / out_sub
    out_dir.mkdir(exist_ok=True)

    for i, f in enumerate(files, 1):
        if prog:
            safe_progress(prog, i, len(files), f.name)

        stem = f.stem

        # ─ rule #1 ───────────────────────────────────────────────
        if "frame_" in stem:
            idx  = stem.find("frame_")
            dash = stem.find("-", idx)          # first '-' AFTER "frame_"
            new_stem = stem if dash == -1 else stem[:dash]

        # ─ rule #2 ───────────────────────────────────────────────
        elif "=" in stem:
            eq   = stem.find("=")
            dash = stem.find("-", eq)           # first '-' AFTER '='
            new_stem = stem if dash == -1 else stem[:dash]

        # ─ fallback ─────────────────────────────────────────────
        else:
            new_stem = stem

        # write copy with same extension
        (out_dir / f"{new_stem}{f.suffix}").write_bytes(f.read_bytes())

    return f"✅  Copied & renamed {len(files)} file(s) ➜ {out_dir}", str(out_dir)

# ───  update-tags helpers  ────────────────────────────────────────────────────
DELIM = ", "

def _clean_tag(t: str) -> str:
    return t.replace("_", " ").replace("\\", "").strip()

def update_tags_in_folder(folder: str, keyword: str) -> str:
    """
    Cleans every *.txt* in `folder`, prepends `keyword` (if not already
    present), and writes results to <folder>/updated/*.txt.
    Returns a summary string.
    """
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(root)

    out_dir = root / "updated"
    out_dir.mkdir(exist_ok=True)

    changed = 0
    for txt in sorted(root.glob("*.txt")):
        raw = txt.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            continue
        tags = [_clean_tag(t) for t in raw.split(DELIM) if t.strip()]
        if keyword not in tags:
            tags.insert(0, keyword)
        new_content = DELIM.join(tags) + "\n"
        (out_dir / txt.name).write_text(new_content, encoding="utf-8")
        changed += 1

    return f"✓ {changed} files processed ➜ '{out_dir.name}' sub-folder."



# ───────────────────────────── UI ─────────────────────────────────
with gr.Blocks(title="Tag-Cleaner Utility") as demo:
    with gr.Tab("Tag Cleaning APP (main)"):
        gr.Markdown(
            """
    # 🏷️ Tag-Cleaner Utility
    
    | Button | Pipeline |
    | ------ | -------- |
    | **Clean Files** | strip `<\\|eot_id\\|>` → split → keep text after `:` → de-dup |
    | **Pad Whitespace Only** | strip `<\\|eot_id\\|>` → replace spaces in each tag with `_` |
    | **E621 Sanitize Tags** | alias/implication → canonical → drop unknown → drop year → drop unwanted categories |
    | **Merge Tag Files** | combine identically-named files from **multiple folders**, de-dup tags, save to **MERGED_TAG_RESULTS** in a user-chosen output folder |
    
    *Multiple input folders* for merging should be given **one per line** or separated by semicolons.  
    All other features work exactly as before.
    """
        )

        # ╌╌ existing single-folder inputs ╌╌
        with gr.Row():
            inp = gr.Textbox(label="Input folder", placeholder=r"C:\datasets\tags", scale=2)
            sub = gr.Textbox(label="Output sub-folder", value="processed", scale=1)
        with gr.Row():
            b_clean = gr.Button("Clean Files", variant="primary")
            b_pad   = gr.Button("Pad Whitespace Only")
            b_e6    = gr.Button("E621 Sanitize Tags", variant="secondary")

        # ╌╌ NEW merge inputs ╌╌
        gr.Markdown("### Merge Tag Files")
        in_dirs_box = gr.Textbox(
            label="Input folders (one per line or ;-separated)",
            placeholder=r"C:\captions\A\nC:\captions\B",
            lines=3,
        )
        out_dir_box = gr.Textbox(
            label="Output base folder",
            placeholder=r"C:\merged_results",
        )
        b_merge = gr.Button("Merge Tag Files", variant="primary")

        gr.Markdown("### Extract Artists (NuMind LLM)")

        artist_inp = gr.Textbox(
            label="Input folder (FA .txt files)", placeholder=r"C:\fa_posts", scale=2)
        artist_outp = gr.Textbox(
            label="Output sub-folder", value="artist_llm", scale=1)
        b_artists = gr.Button("Extract Artists (NuMind)", variant="primary")

        gr.Markdown(
            "### NuMind / Artist-Extraction\n"
            "• Select a **folder of text files** on the left.\n"
            "• Adjust generation settings if you like.\n"
            "• Press **Extract Artists (NuMind)** – results are written to the "
            "`…/ARTIST_TAGS` sub-folder.\n"
        )
        with gr.Row():
            inp_folder   = gr.Textbox(label="Input folder", placeholder="C:\\…")
            out_folder   = gr.Textbox(label="Output folder (optional)")

        # replace the old dropdown construction
        with gr.Row():
            model_drop = gr.Dropdown(
                choices=list(MODELS.keys()),  # ⇐ aliases only
                value=DEFAULT_ALIAS,  # ⇐ default alias
                label="LLM model",
            )

        # DEFAULT_INSTR = (
        #     "PLEASE RETURN YOUR ANSWER IN A \"SINGLE LINE OF TEXT\" (nothing else in the response)\n. The task is defined below:"
        #     "You are/will be given plain text containing information about an artist media post."
        #     "It contains information about the name who made the post as well as who created the actual art for the post."
        #     "The post might be made by the artist that talks about their art piece and possible collaborators (other artist/s) that helped them.\n"
        #     "Or the post might be made by the commissioner of the art, that talks about all the artists involved in creating the art piece.\n"
        #     "### The Rules are as follows:\n"
        #     "• Return a \", \" separated list of artist names and please use \"_\" to pad whitespace inside of individual artist name/s\n. The criteria is as follows:\n"
        #     "• If the user making the post is an artist, include them in the final list.\n"
        #     "• If no artist is mentioned, assume the poster IS the artist.\n"
        #     "• If multiple artists / collabs are indicated, include them all.\n"
        # )
        DEFAULT_INSTR = (
            "**Return ONE line ONLY** – a comma-separated list of artist names.\n"
            "Use underscores for internal spaces (e.g. `John_Doe`).\n"
            "If the uploader is an artist, include them; if none are mentioned, "
            "assume the uploader *is* the artist."
        )
        llm_prompt = gr.Textbox(
            label="LLM Prompt / Instructions",
            lines=8,
            value=DEFAULT_INSTR,
        )
        temperature = gr.Slider(0.0, 2.0, value=0.2, step=0.05,
                                label="Temperature")
        # ────────────── UI construction … (unchanged until here) ──────────────
        do_sample = gr.Checkbox(False, label="Use sampling (do_sample)")

        run_btn = gr.Button("Extract Artists (NuMind)")  # ← already there
        simple_btn = gr.Button("Extract Uploader + Year")  # ← 🆕 line

        # ❶ create the progress helper – NO label, never shown as a component
        prog_bar = gr.Progress()  # or just omit this line entirely

        status = gr.Markdown()  # moved up so we can reuse
        outdir = gr.Textbox(label="Last output folder")
        hf_token_box = gr.Textbox(
            label="🤫 Hugging Face access-token (only needed for gated models)",
            type="password",
            lines=1,
            placeholder="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        )

    # ── ❸ PRIORITISED MERGE (username+year  ⊕  other-tags) ────────────
    gr.Markdown("### Merge Tag-set A  +  Tag-set B  (with priority order)")

    tagA_box = gr.Textbox(
        label="Tag set A folder  (e.g. username+year files)",
        placeholder=r"C:\captions\A_tags",
        scale=2,
    )
    tagB_box = gr.Textbox(
        label="Tag set B folder  (e.g. remaining tags)",
        placeholder=r"C:\captions\B_tags",
        scale=2,
    )
    out_sub_box = gr.Textbox(
        label="Output sub-folder",
        value="merged_tags",
    )

    b_merge_prio = gr.Button("Merge Tags (priority order)", variant="primary")

    # ────────────────────── NEW TAB: metadata patcher ───────────────────────
    with gr.Tab("Fixing Metadata ↦ captions"):
        gr.Markdown(
            "Upload the **posts metadata JSON** and point to the folder that "
            "already contains the individual caption `.txt` files.  \n"
            "The tool injects `<artist>, <year>, ` after the first comma of "
            "each caption (or prepends if no comma is present)."
        )
        meta_uploader = gr.File(label="posts-meta.json", file_types=[".json"])
        caption_dir   = gr.Textbox(label="Caption folder",
                                   placeholder=r"C:\captions")
        b_patch       = gr.Button("Inject artist / year", variant="primary")
    # ─────────────────── Restore original filenames ──────────────────
    with gr.Tab("Restore Filenames"):
        restore_inp = gr.Textbox(
            label="Input folder (up-scaled *.txt files)",
            placeholder=r"C:\upscaled\captions",
            scale=2,
        )
        restore_btn = gr.Button("Restore names → sub-folder", variant="primary")
        restore_stat = gr.Markdown()

    with gr.Tab("Prune Tags (LoRA)"):
        with gr.Row():
            prune_img_path = gr.Textbox(label="Image folder", placeholder="C:\\images")
            prune_txt_path = gr.Textbox(label="Text-tag folder", placeholder="C:\\tags")
        with gr.Row():
            prune_model = gr.Dropdown(choices=list(MODELS.keys()),
                                      value=DEFAULT_ALIAS, label="LLM model")
            prune_token = gr.Textbox(label="HF token (if needed)",
                                     type="password")
        load_btn = gr.Button("▶ Start / Load first")
        img_view = gr.Image(type="pil", label="Preview", height=380)
        with gr.Column():
            chk_char = gr.CheckboxGroup(label="Character")
            chk_spec = gr.CheckboxGroup(label="Species")
            chk_inv = gr.CheckboxGroup(label="Invalid")
            chk_art = gr.CheckboxGroup(label="Artist / Username")
            chk_gen = gr.CheckboxGroup(label="General")
            chk_meta = gr.CheckboxGroup(label="Meta")
            chk_rate = gr.CheckboxGroup(label="Rating")
        CHK_ALL = [chk_char, chk_spec, chk_inv, chk_art,
                   chk_gen, chk_meta, chk_rate]  # handy list

        ok_btn = gr.Button("✓ Apply & next")
        prune_stat = gr.Markdown()

    with gr.Tab("Character pruning"):
        char_src = gr.Textbox(label="Caption folder")
        char_tag = gr.Textbox(label="Trigger character tag (keep only this)")
        b_char = gr.Button("Strip all *other* character names", variant="primary")

    with gr.Tab("Image de-dup"):
        dedup_src = gr.Textbox(label="Image folder")
        b_dedup = gr.Button("Find & move duplicates", variant="primary")

    with gr.Tab("Keyword-Prepended Tag Cleaner"):
        gr.Markdown("### Keyword-Prepended Tag Cleaner")

        with gr.Row():
            keyword_in = gr.Textbox(
                label="Keyword to prepend",
                placeholder="e.g. character_name",
            )
            batch_folder_in = gr.Textbox(
                label="path to image/text folder",
                placeholder="e.g. path to images & text folder",
            )
        with gr.Row():
            update_btn = gr.Button(
                "Update Tags (Add Keyword & Clean)",
                variant="secondary",
            )

        update_out = gr.Textbox(label="Update Summary", lines=3, interactive=False)

    _PRUNE_STATE = gr.State({
        "files": [],  # list[Path] of text files
        "idx": 0,  # current index
        "tok": None,  # cached tokenizer / model
        "model": None,
    })

    # ── helper: classify a single tag using e6_tag_utils data ──────────
    from e6_tag_utils import _CATS  # category id lookup (int → class)


    def _run_update(folder, keyword):
        return update_tags_in_folder(folder, keyword.strip())


    update_btn.click(
        _run_update,
        inputs=[batch_folder_in, keyword_in],
        outputs=[update_out],
    )

    b_char.click(
        lambda p, trig, prog=gr.Progress():
        keep_only_character(Path(p), trig, prog=prog),
        inputs=[char_src, char_tag],
        outputs=[status, outdir],
    )

    b_dedup.click(
        lambda p, prog=gr.Progress():
        dedup_images(Path(p), prog=prog),
        inputs=[dedup_src],
        outputs=[status, outdir],
    )


    # def _class_of(tag: str) -> str:
    #     """Return one of priority classes for sorting."""
    #     tag_lc = tag.lower()
    #     if _CATS.get(tag_lc) == 4:  # character
    #         return "0_char"
    #     elif _CATS.get(tag_lc) == 5:  # species
    #         return "1_spec"
    #     elif _CATS.get(tag_lc) == 6:  # invalid
    #         return "2_inv"
    #     elif re.fullmatch(r"[a-z0-9_]+", tag_lc):  # username / artist
    #         return "3_user"
    #     elif re.fullmatch(r"\d{4}", tag_lc):  # bare year
    #         return "4_year"
    #     elif tag_lc.startswith("meta:"):
    #         return "6_meta"
    #     elif tag_lc.startswith("rating:"):
    #         return "7_rate"
    #     else:
    #         return "5_gen"  # general  (catch-all)
    _CAT_ORDER = ["char", "spec", "inv", "art", "year", "gen", "meta", "rate"]


    # ------------------------------------------------------------------
    #  use e621 category-IDs FIRST, fall back to heuristics
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    #  use e621 category-IDs FIRST, fall back to heuristics
    # ------------------------------------------------------------------
    def _cat_of(tag: str) -> str:
        """
        Map a tag to one of: char • spec • inv • art • gen • meta • rate
        A bare 4-digit year is treated as “meta” for the checkbox UI.
        """
        t = tag.lower().strip()
        cat_id = _CATS.get(t)
        if cat_id == 4:
            return "char"
        elif cat_id == 5:
            return "spec"
        elif cat_id == 6:
            return "inv"
        elif cat_id == 1:
            return "art"

        # ─ fall-back heuristics ────────────────────────────────────────
        if re.fullmatch(r"\d{4}", t):      return "meta"  # ← change here
        if t.startswith("meta:"):          return "meta"
        if t.startswith("rating:"):        return "rate"
        return "gen"


    # ─────────────────────────────────────────────────────────────
    #   keep_only_character()  – strip all other char-tags
    # ─────────────────────────────────────────────────────────────
    def keep_only_character(src_dir: Path,
                            trigger: str,
                            out_sub: str = "CHAR_PRUNED",
                            prog: gr.Progress | None = None) -> tuple[str, str]:
        """
        • `trigger` – the ONE character-tag to keep (e.g.  "my_aniwa")
        • every tag in the <character> bucket that ≠ trigger is removed
        • result is written into  src_dir/out_sub/<same-name>.txt
        """
        trigger = trigger.strip().lower()
        if not trigger:
            return "⚠️  No trigger tag provided.", ""

        src_dir = Path(src_dir).expanduser()
        files = sorted(src_dir.glob("*.txt"))
        if not files:
            return "⚠️  No .txt files found.", ""

        out_dir = src_dir / out_sub
        out_dir.mkdir(exist_ok=True)

        for i, f in enumerate(files, 1):
            if prog: safe_progress(prog, i, len(files), f"Pruning {f.name}")

            tags = [t.strip() for t in f.read_text(encoding="utf-8").split(",") if t.strip()]
            char_tags = [t for t in tags if _cat_of(t) == "0_char"]
            keep_chars = [t for t in char_tags if t.lower() == trigger]
            other_tags = [t for t in tags if _cat_of(t) != "0_char"]

            new_tags = keep_chars + other_tags
            (out_dir / f.name).write_text(", ".join(new_tags), encoding="utf-8")

        return f"✅  Processed {len(files)} file(s) ➜ {out_dir}", str(out_dir)


    # ─────────────────────────────────────────────────────────────
    #   dedup_images()  – hash-based duplicate mover
    # ─────────────────────────────────────────────────────────────
    import hashlib, shutil


    def _file_md5(p: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.md5()
        with p.open("rb") as fh:
            while chunk_data := fh.read(chunk):
                h.update(chunk_data)
        return h.hexdigest()


    def dedup_images(src_dir: Path,
                     dup_sub: str = "duplicates",
                     prog: gr.Progress | None = None) -> tuple[str, str]:
        """
        • Computes an MD5 for *every* image in `src_dir`
        • First occurrence of each hash is kept in place
        • All later duplicates are *moved* into  src_dir/dup_sub
        """
        src_dir = Path(src_dir).expanduser()
        imgs = [p for p in src_dir.iterdir() if p.suffix.lower() in _IMG_EXTS]
        if not imgs:
            return "⚠️  No image files found.", ""

        dup_dir = src_dir / dup_sub
        dup_dir.mkdir(exist_ok=True)

        seen: dict[str, Path] = {}
        moved = 0

        for i, img in enumerate(imgs, 1):
            if prog: safe_progress(prog, i, len(imgs), img.name)
            h = _file_md5(img)
            if h in seen:
                shutil.move(str(img), dup_dir / img.name)
                moved += 1
            else:
                seen[h] = img

        return f"✅  {moved} duplicate(s) moved ➜ {dup_dir}", str(dup_dir)


    def _sort_for_save(tags: list[str]) -> list[str]:
        """Return tags in the mandated output order."""
        buckets = {c: [] for c in _CAT_ORDER}
        for tag in tags:
            buckets[_cat_of(tag)].append(tag)
        return (buckets["char"] + buckets["spec"] +
                buckets["inv"] + buckets["art"] +
                buckets["year"] + buckets["gen"] +
                buckets["meta"] + buckets["rate"])


    def _save_pruned(original_tags: List[str],
                     to_remove: set[str],
                     dst_path: Path):
        kept = [t for t in original_tags if t not in to_remove]
        kept.sort(key=_cat_of)  # stable because Python sort is

        kept = _sort_for_save(kept)
        dst_path.write_text(", ".join(kept), encoding="utf-8")


    def _load_one(pairs, idx, tok, model):
        txt_path, img_path = pairs[idx]
        raw = txt_path.read_text(encoding="utf-8")
        cap_tags = [t.strip() for t in raw.split(",") if t.strip()]

        # LLM → list-of-tags-to-remove
        to_remove = llm_prune(raw, cap_tags, tok, model)

        # split by category
        buckets = {c: [] for c in _CAT_ORDER if c != "year"}
        for tag in to_remove:
            buckets[_cat_of(tag)].append(tag)

        # build gr.update() objects in the same order as CHK_ALL
        updates = [gr.update(choices=buckets["char"], value=buckets["char"]),
                   gr.update(choices=buckets["spec"], value=buckets["spec"]),
                   gr.update(choices=buckets["inv"], value=buckets["inv"]),
                   gr.update(choices=buckets["art"], value=buckets["art"]),
                   gr.update(choices=buckets["gen"], value=buckets["gen"]),
                   gr.update(choices=buckets["meta"], value=buckets["meta"]),
                   gr.update(choices=buckets["rate"], value=buckets["rate"])]

        return (Image.open(img_path), *updates,
                f"File {idx + 1}/{len(pairs)}: {txt_path.name}",
                {"files": pairs, "idx": idx, "tok": tok, "model": model})


    # -----------------------------------------------------------------
    # _load_first  – use the pairing helper
    # -----------------------------------------------------------------
    def _load_first(img_dir, txt_dir, model_alias, hf_token):
        img_dir, txt_dir = Path(img_dir).expanduser(), Path(txt_dir).expanduser()
        pairs = _pair_files(img_dir, txt_dir)
        if not pairs:
            return (gr.update(), gr.update(choices=[]), "⚠ No matching pairs.",
                    {"files": [], "idx": 0, "tok": None, "model": None})

        tok, model = _load_model(model_alias, hf_token or None)
        return _load_one(pairs, 0, tok, model)  # ⬅ only tok/model


    def _apply_next(state, img_dir, txt_dir,
                    char_sel, spec_sel, inv_sel, art_sel,
                    gen_sel, meta_sel, rate_sel):

        pairs, idx, tok, model = state.values()
        if not pairs:
            return (gr.update(), *[gr.update(choices=[]) for _ in CHK_ALL],
                    "✓ Done.", state)

        txt_path, _ = pairs[idx]
        all_tags = [t.strip() for t in txt_path.read_text(encoding="utf-8").split(",") if t.strip()]
        to_drop = set(char_sel + spec_sel + inv_sel + art_sel +
                      gen_sel + meta_sel + rate_sel)

        kept = [t for t in all_tags if t not in to_drop]
        ordered = _sort_for_save(list(dict.fromkeys(kept)))  # de-dup + order

        out_dir = Path(txt_dir).expanduser() / "PRUNED"
        out_dir.mkdir(exist_ok=True)
        (out_dir / txt_path.name).write_text(", ".join(ordered), encoding="utf-8")

        # next file …
        idx += 1
        if idx >= len(pairs):
            return (gr.update(), *[gr.update(choices=[]) for _ in CHK_ALL],
                    "✓ All files finished.",
                    {"files": [], "idx": 0, "tok": tok, "model": model})

        return _load_one(pairs, idx, tok, model)


    # ─── wire the buttons ───────────────────────────────────────────
    load_btn.click(
        _load_first,
        [prune_img_path, prune_txt_path, prune_model, prune_token],
        [img_view, *CHK_ALL, prune_stat, _PRUNE_STATE],  # ← *
    )

    ok_btn.click(
        _apply_next,
        [_PRUNE_STATE, prune_img_path, prune_txt_path, *CHK_ALL],  # ← *
        [img_view, *CHK_ALL, prune_stat, _PRUNE_STATE],  # ← *
    )


    # ── PRIORITISED MERGE ------------------------------------------------
    def merge_priority(a_dir: Path, b_dir: Path, out_sub: str,
                       prog: gr.Progress | None = None) -> tuple[str, str]:
        a_dir, b_dir = Path(a_dir), Path(b_dir)
        if not a_dir.is_dir() or not b_dir.is_dir():
            return "❌  One of the folders does not exist.", ""

        out_dir = (a_dir.parent / out_sub).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        all_names = sorted({*(p.name for p in a_dir.glob("*.txt")),
                            *(p.name for p in b_dir.glob("*.txt"))})
        if not all_names:
            return "⚠️  No .txt files found in either folder.", str(out_dir)

        for i, name in enumerate(all_names, 1):
            if prog:
                safe_progress(prog, i, len(all_names), f"Merging {name}")

            tags: list[str] = []
            for src in (a_dir / name, b_dir / name):
                if src.exists():
                    tags.extend(t.strip() for t in src.read_text(encoding="utf-8").split(", "))

            # de-dup while preserving *first* appearance
            deduped = list(dict.fromkeys(t for t in tags if t))

            # remember original order once, then sort by bucket + original index
            order = {tag: idx for idx, tag in enumerate(deduped)}
            deduped.sort(key=lambda t: (_cat_of(t), order[t]))

            (out_dir / name).write_text(", ".join(deduped), encoding="utf-8")

        return f"✅  Merged {len(all_names)} file(s) ➜ {out_dir}", str(out_dir)


    restore_btn.click(
        lambda p, prog=gr.Progress(): restore_names(Path(p).expanduser(), prog=prog),
        [restore_inp],  # path textbox
        [status, outdir]  # same outputs as before
    )

    # ── wire the new button ───────────────────────────────────────────
    b_merge_prio.click(
        lambda a, b, sub, prog=gr.Progress():
        merge_priority(Path(a).expanduser(),
                       Path(b).expanduser(),
                       sub or "merged_tags",
                       prog),
        inputs=[tagA_box, tagB_box, out_sub_box],
        outputs=[status, outdir],
    )

    simple_btn.click(
        run_uploader_year,
        inputs=[inp_folder, out_folder],  # same two boxes you already have
        outputs=[status, outdir],
    )

    # reuse status/outdir placeholders from main tab
    b_patch.click(
        lambda json_file, cap_dir, prog=gr.Progress():
            inject_from_json(Path(json_file.name), Path(cap_dir).expanduser(), prog),
        inputs=[meta_uploader, caption_dir],
        outputs=status                  # shows the summary string
    )

    # ─ button callbacks ─
    b_clean.click(
        lambda p, s, prog=gr.Progress(): run_over_folder(Path(p).expanduser(), s, basic_clean, prog),
        [inp, sub], [status, outdir]
    )
    b_pad.click(
        lambda p, s, prog=gr.Progress(): run_over_folder(Path(p).expanduser(), s, pad_only, prog),
        [inp, sub], [status, outdir]
    )
    def e6_runner(path, s, prog=gr.Progress()):
        def transform(txt: str) -> str:
            return ", ".join(sanitize_tag_list(_split_tags(_strip_eot(txt))))
        return run_over_folder(Path(path).expanduser(), s, transform, prog)
    b_e6.click(e6_runner, [inp, sub], [status, outdir])

    # merge button callback
    def merge_callback(in_dirs_text, out_path_text, prog=gr.Progress()):
        # split lines / semicolons, expanduser, dedup
        parts = [x.strip() for x in re.split(r"[;\n\r]+", in_dirs_text) if x.strip()]
        dirs  = list(dict.fromkeys(Path(p).expanduser() for p in parts))
        return merge_tag_files(dirs, Path(out_path_text).expanduser(), prog)
    b_merge.click(merge_callback, [in_dirs_box, out_dir_box], [status, outdir])

    def run_artist_llm(folder, out_dir, prompt, temp, sample,
                       model_id, hf_token, prog=gr.Progress()):
        print(f"model_id = {model_id}")

        folder = Path(folder).expanduser()
        files  = sorted(folder.glob("*.txt"))
        if not files:
            return "⚠️  No .txt files found", ""

        out_dir = Path(out_dir or folder / "ARTIST_TAGS").expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(files)

        tok, model = _load_model(model_id, hf_token.strip() or None)

        for i, f in enumerate(files, 1):
            raw = f.read_text(encoding="utf-8")
            uploader, body = extract_body(raw, f.name)       # so the preview shows the file

            artists = get_artists(
                uploader, body,
                model_alias=model_id,
                hf_token=hf_token.strip() or None,   # ← pass on
                temperature=temp,
                do_sample=sample,
                user_prompt=prompt,
                tok=tok,
                model=model,
            )
            (out_dir / f.name).write_text(artists, encoding="utf-8")
            safe_progress(prog, i, total, f"LLM parsed {f.name}")
        return f"✅  Finished – {total} files", str(out_dir)


    run_btn.click(
        run_artist_llm,
        inputs=[inp_folder, out_folder,
                llm_prompt, temperature, do_sample,
                model_drop, hf_token_box],  # +token
        outputs=[status, outdir],
    )

    b_artists.click(
        run_artist_llm,
        inputs=[artist_inp, artist_outp,
                llm_prompt, temperature, do_sample,
                model_drop, hf_token_box],  # +token
        outputs=[status, outdir],
    )

