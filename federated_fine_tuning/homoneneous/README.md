## Environment
First create a Python virtual environment, then install the required packages.

```bash
conda create -n fedalora python=3.8
conda activate fedalora
python -m pip install -r requirements.txt
```

## Dataset
Get the training datasets from [FedDPA](https://github.com/Lydia-yang/FedDPA/).


## Training and Evaluation

The configurations are provided in the training and inference scripts.

```bash
# Training
sh finetune.sh

# Inference
sh inference.sh

# Evaluation
python metric.py
```

## Acknowledgement
The implementation is built on [FedDPA](https://github.com/Lydia-yang/FedDPA/), [FederatedGPT-Shepherd
](https://github.com/JayZhang42/FederatedGPT-Shepherd/). We sincerely thank the authors for their efforts and contributions.