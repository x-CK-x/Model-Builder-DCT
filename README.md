# Data Curation Tool - Model Builder

This Tool is intended to be a placeholder for all bottom-up design to prototype & test new features for the FULL [Dataset Curation Tool](https://github.com/x-CK-x/Dataset-Curation-Tool).
All features here will be integrated into the data curation tool.

![](https://github.com/x-CK-x/Model-Builder-DCT/blob/936dd6d88d2a7e373c4bdd831e6466936872e0ac/GUI_imgs/version_1_gui.png)

## Prerequisites:

- Miniconda (automatically installed by the launch scripts if missing)

## Run Instructions:

1) Download the repository
2) Either: In your windows command line terminal run the following command ``run_local.bat`` from within that folder. OR double-click the ``run_local.bat`` file.
3) Wait until the terminal or pop-up window gives the localhost gradio url & use the browser of your choice to load the GUI with the url.
4) The captioning interface is now included in the main UI and launches with the same command.

-> *please note that the gradio GUI sharing feature (is OFF)* by default.

### Initial Feature Set:

- options to run single & batch configurations for all models listed in the model_registry.json file
- options to select multiple models to run at the same time over the same specified data
- options combine results with models selected
- options to run [classification, grad_cam visualization] tasks
- Captioner tab for single image or batch captioning with prompt customization
  and a dropdown to select different VLM models (JoyCaptioner, LLaVA, Qwen-VL,
  BLIP‑2, InstructBLIP, MiniGPT‑4, Kosmos‑2, OpenFlamingo). Optional Hugging
  Face token input allows downloading gated models.
- supports gpu & cpu run options; all compute options can be enabled in batch mode for higher efficiency (using a forkjoin pool & a thread-safe img queue)

### Features to be implemented:

| Status | Task Description | Priority Order |
|---|---|---|
| [X] | Automatic installation of miniconda (package-manager) for the user, built into the batch script; 1-time install | 1 |
| [X] | Linux/MacOS support; as a shell script | 2 |
| [X] | Captioning Models (multiple VLMs) | 3 |
| [X] | API Access Captioning Models | 4 |
| [X] | Tag Cleaning/Pruning Utility | 5 |
| [X] | Full Pipeline Automation of Data Prep for specific set of model types (Diffusion/LLM LoRAs, classifiers, et al.) | 6 |
| [ ] | Prototype new gradio UI components | 7 |
| [ ] | Merge tool with the Data Curation Tool Project | 8 |

Credits for current model development/options:
- https://huggingface.co/RedRocket/JointTaggerProject
- https://huggingface.co/fancyfeast

### Model-specific notes

- **z3d_convnext**: expects raw 0–255 pixel values. Images are padded to a
  square with a white background using OpenCV, resized to `448×448` and kept in
  BGR channel order. Mean normalization is **not** applied. The accompanying
  `tags.csv` includes a header row (`id,name,category,post_count`) which is
  skipped automatically.
- **eva02_clip_7704**: uses CLIP-style normalization (`mean=[0.485,0.456,0.406]`,
  `std=[0.229,0.224,0.225]`) and requires three placeholder tags (`placeholder0`
  – `placeholder2`) appended after loading `tags.json`.

### VRAM usage

Approximate GPU memory requirements for the built-in models. Values assume 16‑bit weights; quantized checkpoints will need less memory. If the estimate exceeds your single‑GPU VRAM, enable multiple GPUs in the interface so the model can be sharded.

| Classifier model | VRAM (GB) |
|---|---|
| pilot2 | ~6 |
| pilot1 | ~6 |
| z3d_convnext | ~8 |
| eva02_clip_7704 | ~5 |
| eva02_vit_8046 | ~5 |
| efficientnetv2_m_8035 | ~5 |

| Caption model | VRAM (GB) |
|---|---|
| JoyCaptioner | ~14 |
| LLaVA-1.5 | ~14 |
| Qwen-VL | ~16 |
| BLIP2 | ~22 |
| InstructBLIP | ~16 |
| MiniGPT-4 | ~16 |
| Kosmos-2 | ~12 |
| OpenFlamingo | ~18 |

When you select a different caption model in the UI, the previous model is
unloaded automatically. If you still hit CUDA out-of-memory errors after
switching models, the new model likely requires more VRAM than is available on a
single GPU—enable multiple GPUs in the interface or choose a smaller model.
If Kosmos‑2 fails to load with a `KeyError` mentioning `kosmos_2_vision_model`,
update the `transformers` library or use the latest code which patches the
missing configuration automatically.
If Qwen‑VL reports a missing `tiktoken` module, install the optional
`tiktoken` package.
