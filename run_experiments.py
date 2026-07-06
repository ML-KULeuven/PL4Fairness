import sys

from experiments.celeba.celeba import run as celeba_run
from experiments.student.student import run as student_run
from experiments.bias_on_demand.bias_sweep.bias_sweep import run as demand_bias_sweep_run
from experiments.bias_on_demand.assumption_sweep.assumption_sweep import run as demand_assumption_sweep_run

def main(experiment_name, config_path=None):
    run_funcs = {
        "celeba": celeba_run,
        "student": student_run,
        "demand_bias": demand_bias_sweep_run,
        "demand_assumption": demand_assumption_sweep_run,
    }

    assert experiment_name in run_funcs.keys(), f"Experiment name must be one of {tuple(run_funcs.keys())}"

    if config_path is not None:
        run_funcs[experiment_name](config_path)
    else:
        run_funcs[experiment_name]()

if __name__ == "__main__":
    args = sys.argv[1:]
    main(*args)
