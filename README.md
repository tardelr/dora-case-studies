# dora-case-studies

LoRA vs DoRA fine-tuning on Llama-3-8B evaluated on commonsense reasoning benchmarks.

## Setup

```bash
pip install -r requirements.txt
```

Set `HF_TOKEN` in a `.env` file.

## Run

```bash
python train/train.py --config train/config_lora.json --mode all
python train/train.py --config train/config_dora.json --mode all
```

Modes: `train` | `eval` | `all`

## Analysis

Open `analysis.ipynb` and set `RESULTS_ROOT` to your eval output path.
