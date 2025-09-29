## Environment
First create a Python virtual environment, then install the required packages.

```bash
conda create -n alora python=3.10
conda activate alora
python -m pip install -r requirements.txt
```

## Dataset
Get the multi-task NLP dataset from [FedDPA](https://github.com/Lydia-yang/FedDPA/).


## Training and Inference

The configurations are provided in the training and inference scripts.

```bash
# Training
sh run_finetune.sh

# Inference
sh run_inference.sh

# Evaluation
python metric.py
```

## Acknowledgement
The implementation is built on [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/), [FedDPA](https://github.com/Lydia-yang/FedDPA/). We sincerely thank the authors for their efforts and contributions.