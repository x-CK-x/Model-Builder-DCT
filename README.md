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
- JoyCaptioner tab for single image or batch captioning with prompt customization
- supports gpu & cpu run options; all compute options can be enabled in batch mode for higher efficiency (using a forkjoin pool & a thread-safe img queue)

### Features to be implemented:

| Status | Task Description | Priority Order |
|---|---|---|
| [X] | Automatic installation of miniconda (package-manager) for the user, built into the batch script; 1-time install | 1 |
| [X] | Linux/MacOS support; as a shell script | 2 |
| [X] | Captioning Models (JoyCaptioner) | 3 |
| [X] | API Access Captioning Models | 4 |
| [ ] | Tag Cleaning/Pruning Utility | 5 |
| [ ] | Full Pipeline Automation of Data Prep for specific set of model types (Diffusion/LLM LoRAs, classifiers, et al.) | 6 |
| [ ] | Prototype new gradio UI components | 7 |
| [ ] | Merge tool with the Data Curation Tool Project | 8 |

Credits for current model development/options:
- https://huggingface.co/RedRocket/JointTaggerProject
- https://huggingface.co/fancyfeast
