"""
artist_llm.py  –  simple wrapper around a *choice* of HF instruct models
No Flash-Attention, no Triton, no stubs – runs on Windows CUDA out-of-box.

    DEFAULT_INSTR = (
        "PLEASE RETURN YOUR ANSWER IN A \"SINGLE LINE OF TEXT\" (nothing else in the response)\n. The task is defined below:"
        "You are/will be given plain text containing information about an artist media post."
        "It contains information about the name who made the post as well as who created the actual art for the post."
        "The post might be made by the artist that talks about their art piece and possible collaborators (other artist/s) that helped them.\n"
        "Or the post might be made by the commissioner of the art, that talks about all the artists involved in creating the art piece.\n"
        "### The Rules are as follows:\n"
        "• Return a \", \" separated list of artist names and please use \"_\" to pad whitespace inside of individual artist name/s\n. The criteria is as follows:\n"
        "• If the user making the post is an artist, include them in the final list.\n"
        "• If no artist is mentioned, assume the poster IS the artist.\n"
        "• If multiple artists / collabs are indicated, include them all.\n"
    )

"""
from __future__ import annotations
# put this at the VERY top of artist_llm.py
import os
os.environ["CUDA_VISIBLE_DEVICES"]          = "0,1"      # be explicit
os.environ["PYTORCH_CUDA_ALLOC_CONF"]       = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_HF_XET"]         = "1"        # your TLS fix


# (rest of your imports)
from pathlib import Path
import torch, json, re, functools

import functools, json, re
from pathlib import Path
from typing import Literal, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# -------------------------------------------------------------------
# 🧠  available models ------------------------------------------------
# ── available models ─────────────────────────────────────────────
MODELS: dict[str, str] = {
    # human-readable alias  →  HF model id
    # alias       →  HF model id                                  (token?)
    "phi3":    "microsoft/Phi-3-mini-4k-instruct",               # public
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",             # gated ✅
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",   #👈 REMOVE this line
    "gemma":     "google/gemma-7b-it",          # ← NEW – text-only
}

DEFAULT_ALIAS = "phi3"          # make sure this is still in MODELS

# ─── at the very top of artist_llm.py (keep the old imports) ──────
_DT_RX  = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")  # timestamp
_YEAR_RX = re.compile(r"^\d{4}$")                            # 4-digit year
_WS_RX  = re.compile(r"^[,\s]+")                             # ,,␠␠␠

def _debug_preview(fname: str, body: str, n: int = 160) -> None:
    head = body[:n//2].replace("\n", "⏎")
    tail = body[-n//2:].replace("\n", "⏎")
    print(f"[DBG] {fname:>30} │ {head} … {tail}")

# ─── NEW unified parser ───────────────────────────────────────────
def extract_body(raw: str, fname: str = "<memory>") -> tuple[str, str]:
    """
    Returns (uploader_username, cleaned_body).

    The header layout we expect **after the first comma** is

        <post_title>, <uploader_username>, <year>, …

    • We ignore every leading ',' or whitespace after the very first comma.
    • We split on ', ' and locate the first 4-digit YEAR token.
    • The element immediately **before** that year is taken as *uploader*.
    • *Body*  = everything that follows the year **up to** the timestamp
      (the yyyy-mm-dd hh:mm:ss string), which is excluded.
    """
    # ── ① uploader + year -----------------------------------------
    first_comma = raw.find(",")
    start = first_comma + 1 if first_comma != -1 else 0

    # eat any sequence of   ,  / spaces / tabs / newlines
    m_ws  = _WS_RX.match(raw, pos=start)
    start = m_ws.end() if m_ws else start

    # anything before timestamp belongs to “header”
    m_dt  = _DT_RX.search(raw, pos=start)
    header_end = m_dt.start() if m_dt else len(raw)
    header     = raw[start:header_end]

    parts = [p.strip() for p in header.split(", ") if p.strip()]
    year_idx = next((k for k, p in enumerate(parts) if _YEAR_RX.match(p)), None)

    # fallback safety
    if year_idx is None or year_idx == 0:
        uploader = ""
    else:
        uploader = parts[year_idx - 1].lower().replace(" ", "_")

    # ── ② body text ----------------------------------------------
    body_start = header_end
    body_end   = m_dt.start() if m_dt else len(raw)
    body       = raw[body_start:body_end].strip(", ").strip()

    _debug_preview(fname, body)      # DEBUG preview shown in console
    return uploader, body






def build_prompt(instructions: str, uploader: str, body: str) -> str:
    """
    • If the user-supplied prompt contains <UPLOADER> or <BODY>, substitute.
    • Otherwise append a tiny header block the model can key off.
    """
    if "<UPLOADER>" in instructions or "<BODY>" in instructions \
       or "<POSTER>"   in instructions:          # ← legacy fallback
        return (instructions
                .replace("<UPLOADER>", uploader)
                .replace("<POSTER>",   uploader)  # old alias
                .replace("<BODY>",     body))
    # minimal, structured fallback
    return (f"{instructions.rstrip()}\n"
            f"### Uploader:\n{uploader}\n"
            f"### Body:\n{body}\n"
            "<|output|>\n")


#  artist_llm.py  –– twin-GPU loader for Gemma-7B-IT (and friends)
# -----------------------------------------------------------------
# ────────────────────────────────────────────────────────────
#   artist_llm.py – single-function patch
#   drop this in and delete every previous _load_model()
# ────────────────────────────────────────────────────────────
import functools, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM

VRAM_LIMIT = 19                      # soft cap / GPU (24 GB cards)

@functools.lru_cache(maxsize=1)
def _load_model(alias: str, hf_token: str | None = None):
    print(f"alias = {alias}")
    print(f"model_id = {MODELS[alias]}")

    hf_token = hf_token or None          # <- ADD THIS
    model_id = MODELS[alias]

    # one-liner login (does nothing if token is None or already saved)
    if hf_token:
        from huggingface_hub import login
        login(hf_token, add_to_git_credential=True)

    max_mem = {i: f"{VRAM_LIMIT}GiB" for i in range(torch.cuda.device_count())}

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map            = "auto",            # <- let HF/Accelerate split
        max_memory            = max_mem,
        torch_dtype           = torch.float16,     # 3090 = fp16 only
        low_cpu_mem_usage     = True,              # stream load
        use_safetensors       = True,
    )

    tok = AutoTokenizer.from_pretrained(model_id, use_safetensors=True)
    model.generation_config.pad_token_id = tok.eos_token_id
    return tok, model




def get_artists(
    uploader: str,
    body: str,
    *,
    model_alias: str = DEFAULT_ALIAS,
    hf_token:   str | None = None,
    temperature: float = 0.2,
    do_sample:   bool  = False,
    user_prompt: str   = "",
    tok=None,
    model=None,
) -> str:
    # ── ① lazy-load model once ───────────────────────────────────────────
    if model is None or tok is None:
        tok, model = _load_model(model_alias, hf_token)

    # ── ② build prompt ──────────────────────────────────────────────────
    DEFAULT_INSTR = (
        "**Return ONE line ONLY** – a comma-separated list of artist names.\n"
        "Use underscores for internal spaces (e.g. `John_Doe`).\n"
        "If the uploader is an artist, include them; if none are mentioned, "
        "assume the uploader *is* the artist."
    )
    prompt = build_prompt(user_prompt or DEFAULT_INSTR, uploader, body)

    # ── ③ generation kwargs ─────────────────────────────────────────────
    gen_cfg = dict(
        max_new_tokens = 64,
        do_sample      = do_sample,
        temperature    = temperature if do_sample else None,
        use_cache      = False,                 # avoids GPU-0 spill-over
    )
    if model_alias.startswith("phi3"):
        model.config.use_cache = False          # safety for old Phi-3

    # ── ④ run the model ─────────────────────────────────────────────────
    with torch.inference_mode():
        out = model.generate(
            **tok(prompt, return_tensors="pt").to(model.device),
            **gen_cfg,
        )

    # ── ⑤ post-process ──────────────────────────────────────────────────
    resp = tok.decode(out[0], skip_special_tokens=True)
    # grab only the part after last marker if user left <|output|>
    resp = resp.split("<|output|>")[-1].strip()

    # simple de-dup keeping order
    artists = [a.strip() for a in resp.split(",") if a.strip()]
    return ", ".join(dict.fromkeys(artists))
