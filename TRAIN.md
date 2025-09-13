# Installation 

**This guide will use uv as the main way to manage dependencies and virtual environments.**

System where this has been tested:

- Opensuse tumbleweed

## Install dependencies 

```bash 
uv sync
```

## Train process setup

### Train dependencies

Not all dependencies are needed for training, so if you want to install only the needed ones, you can use the following command:
```bash
```txt
albumentations
hydra-core
matplotlib
pandas
pillow
scikit-learn
scipy
torch
torchvision
tqdm
wandb
```
