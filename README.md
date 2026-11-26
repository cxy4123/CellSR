# CellSR: selective parameter sharing across symbolic equations

Research code for *When to share a physical parameter across heterogeneous symbolic equations* by Xiongheng Bian, Xiangyu Cui, Ma Feng and Xiaoyan Shen.

CellSR fits separate local equations and evaluates a proposed equality between coefficient locations on held-out data. This repository contains the coefficient-fitting implementation, the simulation experiments and archived simulation outputs, gait analysis scripts, cohort-level gait results, and the three manuscript figures.

## Reproduce the simulation results

Run from the repository root in a Python 3.12 environment:

```sh
python -m pip install -r handoff/requirements.txt
python -m pytest -q tests
python papers/CellSR_MLST_Chinese/revision_zh/simulation_table.py
python handoff/strong_conflict_table.py
```

The first analysis refits four estimators on the archived expressions for 256 tasks. In the equal-coefficient setting, independent fitting and CellSR yield target MAEs of 0.1490 and 0.0842. The second analysis checks the 128 strong-conflict tasks and calculates the paired bootstrap interval.

To regenerate the simulations, run these scripts in order. They write into the corresponding directories under `benchmark_outputs/cellsr1_revision/`.

```sh
python experiments/cellsr1_revision/search_path_ablation.py
python experiments/cellsr1_revision/coefficient_gap.py
python experiments/cellsr1_revision/gate_sensitivity.py
python experiments/cellsr1_revision/sample_noise.py
python experiments/cellsr1_revision/soft_decisions.py
```

## Walking analyses

The Horst and Antwerp datasets are available from their publishers: [Horst, dataset v3](https://doi.org/10.17632/svx74xcrjr.3) and [Van Criekinge et al., collection v1](https://doi.org/10.6084/m9.figshare.c.6503791.v1). Obtain source recordings and participant reference data from those repositories under the respective terms. Raw recordings, individual windows, reference masses and individual predictions are not distributed here.

Horst window extraction and scoring are in `experiments/cellsr1_revision/external_horst_windows.py` and `external_pair_compare.py`. Antwerp extraction, matching and scoring use `matched_antwerp_windows.py`, `matched_false_share.py` and `matched_antwerp_mass.py`. Raw C3D extraction also requires a platform-compatible ezc3d installation and the files at the paths used by the scripts. Participant-level bootstrap calculations and the individual trajectories in Figure 3a require the local derived records. The included comparison tables can be read directly without downloading recordings.

## Locate reported outputs

| Manuscript result | File or directory |
|---|---|
| Table 1a, four estimators | `papers/CellSR_MLST_Chinese/revision_zh/simulation_table.csv` |
| Table 1b, strong conflict | `benchmark_outputs/cellsr1_revision/soft_decisions/strong_conflict_table.csv` |
| Expression search and recovery | `benchmark_outputs/cellsr1_revision/search_path_classsr_v1/` |
| Coefficient gaps and sensitivity | `benchmark_outputs/cellsr1_revision/{coefficient_gap,gate_sensitivity,sample_noise,soft_decisions}/` |
| Walking cohorts | `benchmark_outputs/cellsr1_revision/external_case_comparison_v1/gait/balanced.csv` and `benchmark_outputs/cellsr1_revision/matched_false_share/comparison.csv` |
| Final figures | `papers/CellSR_Neurocomputing/figures/` |
| Figure source | `papers/CellSR_MLST_Chinese/framework_v2/make_figures.py` |

Corresponding author: Xiongheng Bian (bxh@ntu.edu.cn).
