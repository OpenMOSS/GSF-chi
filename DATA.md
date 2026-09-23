# Data preparation and provenance

No molecular records, spectra, labels, cached graphs, conformers, or trained
weights are stored in this repository. Obtain every dataset from its original
provider and keep its original license and citation. The code expects the local
layout below; symbolic links are acceptable.

## Expected layout

```text
data/
  RS_train.pkl
  RS_validation.pkl
  RS_test.pkl
  ranking_train.pkl
  ranking_validation.pkl
  ranking_test.pkl

  axial_650.xlsx
  ecd_axial_index_split.pkl
  hct_ecd_axial.pkl
  hct_ecd_axial_res.pkl
  optical_rotation_589nm.csv

  central_ecd_raw/
    ecd_column_charity_new_smiles.npy
    extracted/ECD/
      500ECD/data/*.csv
      501-2000ECD/data/*.csv
      2k-6kECD/data/*.csv
      6k-8kECD/data/*.csv
      8k-11kECD/data/*.csv

  external/rota/
    RotA.pkl
    RotA.xlsx

  moleculenet/
    BBBP.csv
    bace.csv
    clintox.csv.gz
    sider.csv.gz
    SAMPL.csv
```

Only the first three groups are needed for the central and ACMP headline
benchmarks. RotA and MoleculeNet are optional generalization evaluations.

## 1. Central R/S and enantiomer ranking

Source: the exact ChIRo task splits linked from the
[ChIRo repository](https://github.com/keiradams/ChIRo) and hosted in the
[official Figshare share](https://figshare.com/s/e23be65a884ce7fc8543).

The source archive used for our local reconstruction had:

```text
size    2,741,008,501 bytes
sha256  47c6bc5bc7149752eb20a95705da81774c6df9dff0ace56abb64919098d49800
```

After extraction, link or copy these six files to the short names expected by
the loaders:

| Local name | Source filename | Rows / pairs |
|---|---|---:|
| `RS_train.pkl` | `train_RS_classification_enantiomers_MOL_326865_55084_27542.pkl` | 326,865 rows |
| `RS_validation.pkl` | `validation_RS_classification_enantiomers_MOL_70099_11748_5874.pkl` | 70,099 rows |
| `RS_test.pkl` | `test_RS_classification_enantiomers_MOL_69719_11680_5840.pkl` | 69,719 rows |
| `ranking_train.pkl` | `train_small_enantiomers_stable_full_screen_docking_MOL_margin3_234622_48384_24192.pkl` | 24,192 pairs |
| `ranking_validation.pkl` | `validation_small_enantiomers_stable_full_screen_docking_MOL_margin3_49878_10368_5184.pkl` | 5,184 pairs |
| `ranking_test.pkl` | `test_small_enantiomers_stable_full_screen_docking_MOL_margin3_50571_10368_5184.pkl` | 5,184 pairs |

Required columns are:

- R/S: `rdkit_mol_cistrans_stereo`, `RS_label_binary`;
- Ranking: `rdkit_mol_cistrans_stereo`, `ID`, `top_score`.

The upstream train/validation/test partitions are used unchanged. The audit in
our loader also verifies that base-molecule identities do not overlap across
partitions.

## 2. Central ECD (public CMCDS reconstruction)

Source: [ECDFormer_Datasets](https://huggingface.co/datasets/OzymandisLi/ECDFormer_Datasets),
revision `cfd955a686e9c24a1ca65e9c26dc69f62ee41bd5`.

Install Git LFS and retrieve the `ECD/` directory:

```bash
git lfs install
git clone https://huggingface.co/datasets/OzymandisLi/ECDFormer_Datasets
cd ECDFormer_Datasets/ECD
cat part_aa part_ab | tar -xzvf -
```

Copy or link:

```text
ECD/ecd_column_charity_new_smiles.npy
    -> data/central_ecd_raw/ecd_column_charity_new_smiles.npy

the extracted ECD directory
    -> data/central_ecd_raw/extracted/ECD/
```

Expected source-object checksums:

| Object | Bytes | SHA-256 |
|---|---:|---|
| `ecd_column_charity_new_smiles.npy` | 342,684,749 | `6f04d36d99ef6b2a263f7767fe24777d299eb789dc65832847e74b5aeccd6dec` |
| `part_aa` | 1,073,741,824 | `e78dd4ce318bd89003dca0922464b689b07ca643c27983d114d46b2bb333768c` |
| `part_ab` | 310,883,131 | `ddd6b6c6250b8d7fadda7bc610fd2e178575b3bee703a15e84ec76750ce8df14` |

The archive exposes 10,335 measured spectra. The deterministic builder in
`src/gsf_chi/data/central_ecd.py` pairs each observed spectrum with its
opposite stereoisomer through `hand_id`, complements the peak signs, and removes
four allene pairs without exactly one assigned tetrahedral center. The resulting
public benchmark has 10,331 enantiomer pairs (20,662 molecular samples).

The primary split is generated at **pair level** with Python RNG seed 42:

| Partition | Pairs | Molecular samples |
|---|---:|---:|
| train | 8,264 | 16,528 |
| validation | 1,033 | 2,066 |
| test | 1,034 | 2,068 |

Build the cache without training:

```bash
python scripts/prepare_central_ecd.py \
  --raw-root data/central_ecd_raw/extracted/ECD \
  --graph-path data/central_ecd_raw/ecd_column_charity_new_smiles.npy \
  --output data/central_ecd_raw/gsf_central_ecd_cache.pkl
```

This public 20,662-sample reconstruction is not the unreleased 22,182-molecule
processed pickle cited by ChiDeK. Cross-method comparisons in our matched table
retrain every method on the public reconstruction with the same split and
evaluator.

## 3. ACMP axial chirality

Source: [ChiDeK](https://github.com/Meteor-han/ChiDeK), revision
`671ed8a71cf0e53116f181260a3f28dfe3d21851`. Copy the five files from its
`data/` directory into this repository's `data/` directory.

| File | Bytes | SHA-256 |
|---|---:|---|
| `axial_650.xlsx` | 35,054 | `8c4fc57a4272813029b9b0fd6ecfb5e032d412d606804b26c8a24ba88ec295f0` |
| `ecd_axial_index_split.pkl` | 3,387 | `2d6ed485fe14bc238702006b41d699342618bbab0136eebcfc9858b6aa43aa14` |
| `hct_ecd_axial.pkl` | 1,899,212 | `db1d9369e0a77b2a9cf257e5470d226d9c22c86d4b08f3fa7cbc14d16841933e` |
| `hct_ecd_axial_res.pkl` | 649,819 | `b6545e2c4a99b50926dfb0e7f6d0cfe8d898ca919176fe5303e30c9d3339ba24` |
| `optical_rotation_589nm.csv` | 13,879 | `6a70b6298209be02833f4a15030863124efc20f5a523582b8b5f9936d854e935` |

`ecd_axial_index_split.pkl` is used without modification and contains 952 / 120 /
120 molecular samples for train / validation / test (476 / 60 / 60 complete
enantiomer pairs). Both axial Rotation and axial ECD use this same split.

`hct_ecd_axial_res.pkl` contains the released ChiralFinder annotations. Passing
`--annotation-source automatic` uses label-independent automatic detections;
the default `curated` mode uses the benchmark annotations.

## 4. RotA conformer evaluation (optional)

Source: [ChiralFinder](https://github.com/Meteor-han/chiralfinder), revision
`d191f81cf2bef8a81d2dcb8f0652cff50e527bff`. The release describes 3,140
conformers for 650 axially chiral molecules.

Copy `data/RotA.pkl` and `data/RotA.xlsx` from ChiralFinder to
`data/external/rota/`. Expected checksums are:

| File | Bytes | SHA-256 |
|---|---:|---|
| `RotA.pkl` | 1,773,015 | `20a1f7ba8c3d30329c712600d564765e0a8eb3c493f2e0bf16d82aa42439d212` |
| `RotA.xlsx` | 54,154 | `141ae5281c034f8d08b900454fc387556b5c8702b3e2347d7141ddbd69c4daff` |

Prepare the intersection with the locked ACMP test identities:

```bash
git clone https://github.com/Meteor-han/chiralfinder data/external/chiralfinder
python scripts/prepare_rota.py \
  --data-dir data \
  --rota-pkl data/external/rota/RotA.pkl \
  --chiralfinder-root data/external/chiralfinder \
  --output outputs/rota_acmp_test.pkl
```

This is a conformer-shift evaluation on ACMP identities, not a new labeled
training split.

## 5. MoleculeNet downstream tasks (optional)

The downstream script expects the canonical MoleculeNet CSV files distributed
by [DeepChem](https://deepchem.io/) under `data/moleculenet/`:

| Task | Filename | Metric |
|---|---|---|
| BBBP | `BBBP.csv` | ROC-AUC |
| BACE | `bace.csv` | ROC-AUC |
| ClinTox | `clintox.csv.gz` | mean ROC-AUC |
| SIDER | `sider.csv.gz` | mean ROC-AUC |
| FreeSolv | `SAMPL.csv` | RMSE |

The preparation stage generates deterministic conformers and a scaffold split;
generated graph caches remain under `outputs/` and are not committed.

```bash
gsf-chi moleculenet \
  --prepare --train --summarize --seeds 0 1 2 --device cuda
```

## Metrics and label conventions

- **R/S** and **Rotation**: sample accuracy.
- **Ranking**: accuracy of the predicted ordering within each enantiomer pair.
- **Position**: RMSE over the ordered peak-position targets.
- **Number**: RMSE of the predicted number of peaks.
- **Symbol**: the released evaluator scores real peaks for nonempty spectra.
  Empty spectra contribute a fixed correct count equal to the task slot count.
- Under mirror inversion, Number and Position are treated as even targets;
  Rotation and Symbol are odd targets. The projected ECD readout enforces this
  contract exactly at the logit level.

Run `python scripts/check_data.py --dataset core` after preparation. The checker
reports missing files, checksum mismatches, split sizes, and central-table row
counts without writing or uploading any molecular data.
