# Data Curation Tool - Model Builder

This Tool is intended to be a placeholder for all bottom-up design to prototype & test new features for the FULL [Dataset Curation Tool](https://github.com/x-CK-x/Dataset-Curation-Tool).
All features here will be integrated into the data curation tool.

![](https://github.com/x-CK-x/Model-Builder-DCT/blob/936dd6d88d2a7e373c4bdd831e6466936872e0ac/GUI_imgs/version_1_gui.png)

## Prerequisites:

- https://www.anaconda.com/docs/getting-started/miniconda/install

## Run Instructions:

1) Download the repository
2) Either: In your windows command line terminal run the following command ``run_local.bat`` from within that folder. OR double-click the ``run_local.bat`` file.
3) Wait until the terminal or pop-up window gives the localhost gradio url & use the browser of your choice to load the GUI with the url.

-> *please note that the gradio GUI sharing feature (is OFF)* by default.

### Initial Feature Set:

- options to run single & batch configurations for all models listed in the model_registry.json file
- options to select multiple models to run at the same time over the same specified data
- options combine results with models selected
- options to run [classification, grad_cam visualization] tasks
- supports gpu & cpu run options; all compute options can be enabled in batch mode for higher efficiency (using a forkjoin pool & a thread-safe img queue)

### Features to be implemented:

- new model list **[TBD]**
- new UI features **[TBD]**
- new custom gradio components **[TBD]**
- new data management features **[TBD]**
- new data analysis tools **[TBD]**
- new manual & automatic data augmentation & data pipeline features **[TBD]**
- Linux/MacOS support
- automatic installation of miniconda for the user

Credits for model development:
- https://huggingface.co/RedRocket/JointTaggerProject
- 
