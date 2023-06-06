# Repository for the paper [*Radio Galaxy Zoo: Building a multi-purpose foundation model for radio astronomy with self-supervised learning*](https://arxiv.org/abs/2305.16127)

# Instructions for use

## Installation
- Clone this repo.
- Checkout the `reproduce` branch. (**IMPORTANT**)
- Create a new python venv with python 3.6+ 
- Activate your virtual environment.
- Install packages by running `pip install -r requirements.txt` (make sure you are in the root of the project).
- Navigate to the parent directory of this repo and locally install this repo as a package by running `pip install -e byol`.
- Clone the `main` branch of the [AstroAugmentations repository](https://github.com/mb010/AstroAugmentations) repository and install locally by using `pip install -e AstroAugmentations`.
- Make sure you have a gpu available with enough memory to load a ResNet-18 model. Most modern laptops with a dedicated graphics card should be OK. You may need to install some drivers to access the card.

## Fine-tuning
- We provide a pre-trained checkpoint which you can download [here](https://www.dropbox.com/s/3ai64rgtzeim682/byol.ckpt?dl=0).
- This checkpoint is the model with optimized hyper-parameters which achieves ~98% accuracy when fine-tuned on all MiraBest Confident training data and evaluated on the MiraBest Confident test set.
- Place `byol.ckpt` into the main directory of the project (same directory as `finetuning.py`).
- Specify settings in `finetune.yml`, **making sure that** `run_id: 'none'`. 

## Pre-training
The RGZ DR1 data-set is currently proprietary, but will be released in due course, at which point we can release the data-set used to pre-train. All the code required for pre-training is available to view in this repository in `train.py` and `models.py`.
