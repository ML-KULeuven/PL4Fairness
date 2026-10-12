# ProbLog4Fairness

![Static Badge](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white)

Code for the paper **"ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias"**, accepted at AAAI 2026.

## Requirements

- Python >= 3.12
- For a list of required packages, see `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ML-KULeuven/PL4Fairness.git
   cd PL4Fairness
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Structure

```
experiments/
  bias_on_demand/    # Synthetic bias experiments
  celeba/            # CelebA dataset experiments
  student/           # Student dataset experiments
res/data/            # Directory for dataset files
src/                 # Core method implementation
demo.ipynb           # Notebook with demo for label bias
```

## Running the Demo Notebook

`demo.ipynb` shows how to use ProbLog4Fairness in practice. On the student dataset, it wraps an ordinary
PyTorch classifier in a `LabelBiasModule`, trains it on biased labels, and compares the result with a
classifier trained directly on the biased labels and one trained on the fair labels.

After completing the steps under Installation, make sure Jupyter is installed.
Then from the repository root, open the notebook and run the cells:
```bash
jupyter lab demo.ipynb
```

The notebook uses `LabelBiasModule`, which can be found in `src/fairness_module.py` together with `MeasurementBiasModule` and `HistoricalBiasModule`.
These modules are the easiest way to get started with ProbLog4Fairness.

## Running Experiments

Experiments are run via `run_experiments.py`, which accepts an experiment name and a path to a YAML config file:

```bash
python run_experiments.py <experiment_name> <config_path>
```

Available experiment names:

| Name | Description |
|------|-------------|
| `celeba` | CelebA dataset experiment |
| `student` | Student dataset experiment |
| `demand_bias` | Synthetic bias sweep (bias on demand) |
| `demand_assumption` | Synthetic assumption sweep (bias on demand) |

Examples:

```bash
python run_experiments.py celeba experiments/celeba/configs/celeba.yaml
python run_experiments.py student experiments/student/configs/student.yaml
python run_experiments.py demand_bias experiments/bias_on_demand/bias_sweep/configs/label.yaml
```

Alternatively, call an experiment's `run(config_path)` function directly:

```python
from experiments.student.student import run
run("experiments/student/configs/student.yaml")
```

## Citation

If you use this code, consider citing us:

```bibtex
@article{adriaensen2026pl4fairness,
    title     = {ProbLog4Fairness: A Neurosymbolic Approach to Modeling and Mitigating Bias},
    author    = {Adriaensen, Rik and Van Praet, Lucas and Bekker, Jessa and Manhaeve, Robin and Delobelle, Pieter and Buyl, Maarten},
    year      = {2026},
    journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
    volume    = {40},
    number    = {24},
    pages     = {19542--19550},
    url       = {https://ojs.aaai.org/index.php/AAAI/article/view/39033},
    DOI       = {10.1609/aaai.v40i24.39033}
}
```

## Data

The `student` experiment uses a copy of [this](https://www.kaggle.com/datasets/daphnelenders/performance-vs-predicted-performance) dataset introduced by Lenders and Calders
(2023) provided without modification in `res/data/student` under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license.