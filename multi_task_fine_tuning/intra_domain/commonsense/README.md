## Environment
First create a Python virtual environment, then install the required packages.

```bash
conda create -n alora python=3.10
conda activate alora
python -m pip install -r requirements.txt
```

## Dataset
Get the commonsense and math reasoning benchmarks from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/).


## Training and Evaluation

The configurations are provided in the training and inference scripts.

```bash
# Training
sh run_commonsense.sh

# Evaluation
sh eval_commonsense.sh
```

## Acknowledgement
The implementation is built on [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters/), [MoSLoRA](https://github.com/wutaiqiang/MoSLoRA/). We sincerely thank the authors for their efforts and contributions.