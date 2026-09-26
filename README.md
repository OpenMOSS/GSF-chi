<div align="center">

# GSF-χ

### Global Stereochemical Fields for Chiral Graph Transformers

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[Installation](#installation) · [Quick Start](#quick-start) · [Experiments](#experiments) · [Data](DATA.md)

</div>

## Introduction

GSF-χ is a Graph Transformer for chiral molecules. It represents central and
axial stereogenic units as fields over molecular atoms and uses handedness to
control relative query–key rotations. Its ECD readout separates mirror-even
peak counts and positions from mirror-odd peak signs.

This repository contains the model, data preparation scripts, and configurations
for R/S classification, enantiomer ranking, optical rotation, and central and
axial ECD prediction.

## Installation

Requires Python 3.10 or later.

Use **Download ZIP** on this page, extract the archive, and open a terminal in
the extracted repository directory.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Install a PyTorch build compatible with your CUDA runtime for GPU training.
Rebuilding RotA stereogenic-unit annotations also needs `pip install -e '.[automatic]'`.

## Quick Start

Follow [DATA.md](DATA.md) to obtain the datasets and place them under `data/`.
Run commands from the repository root; paths in configurations are relative to
the working directory.

```bash
python scripts/check_data.py --dataset core
```

Build the central-ECD cache with `python scripts/prepare_central_ecd.py`.
The R/S and Ranking tasks prepare their caches on first use.

## Experiments

Each task has a configuration in [configs/paper](configs/paper).

```bash
gsf-chi axial_rotation --config configs/paper/axial_rotation.json
gsf-chi axial_ecd      --config configs/paper/axial_ecd.json
gsf-chi rs             --config configs/paper/rs.json
gsf-chi ranking        --config configs/paper/ranking.json
gsf-chi central_ecd    --config configs/paper/central_ecd.json
```

`python -m gsf_chi` is equivalent to `gsf-chi`. Add `--dry-run` to inspect the
resolved arguments. Command-line options override configuration values:

```bash
gsf-chi axial_ecd --config configs/paper/axial_ecd.json --device cpu --dry-run
```

## Repository Structure

| File | Purpose |
|---|---|
| [`attention.py`](src/gsf_chi/attention.py) | Global field, pair gates, and Chiral-RoPE correction |
| [`model.py`](src/gsf_chi/model.py) | Graph Transformer and molecular readout |
| [`ecd.py`](src/gsf_chi/ecd.py) | Shared encoder and parity-projected ECD heads |
| [`ecd_metrics.py`](src/gsf_chi/ecd_metrics.py) | ECD losses, decoders, and evaluation |
| [`data/`](src/gsf_chi/data) | Molecular features and benchmark loaders |
| [`tasks/`](src/gsf_chi/tasks) | Training and checkpoint selection |
| [`controls.py`](src/gsf_chi/controls.py) | Interaction masks and reduced-supervision protocols |

[configs/ablations](configs/ablations) contains the component and representation
controls; [configs/controls](configs/controls) contains equal-support and
one-enantiomer experiments. [configs/matched](configs/matched) records the
separate replication configurations.

For MoleculeNet, run `gsf-chi moleculenet --help`. RotA preparation and evaluation
are in `scripts/prepare_rota.py` and `scripts/evaluate_rota.py`.

## Tests

```bash
pytest -q -m 'not data'
# With ACMP, central ECD and Ranking data prepared:
pytest -q
python scripts/audit_reflection.py --help
```

Tests cover rotation inversion, atom and unit permutations, achiral reduction,
mirror parity, gradient isolation, and the support and supervision controls.

## Acknowledgements

Datasets and comparison methods come from [ChiDeK](https://github.com/Meteor-han/ChiDeK),
[ChIRo](https://github.com/keiradams/ChIRo),
[ECDFormer](https://huggingface.co/datasets/OzymandisLi/ECDFormer_Datasets), and
[ChiralFinder](https://github.com/Meteor-han/chiralfinder).
See [DATA.md](DATA.md) for revisions and checksums. Dataset files and trained
weights are obtained separately.

## License

The code is released under the [MIT license](LICENSE).

## Citation

If you use GSF-chi in your research, please cite
[GSF-χ: Global Stereochemical Fields for Chiral Graph Transformers](https://arxiv.org/abs/2609.12532):

```bibtex
@misc{xie2026gsfchi,
  title         = {{GSF-$\chi$}: Global Stereochemical Fields for Chiral Graph Transformers},
  author        = {Jiaqing Xie and Yuxin Wang and Xipeng Qiu},
  year          = {2026},
  eprint        = {2609.12532},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2609.12532}
}
```
