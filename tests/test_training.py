"""Exercise complete training steps and checkpoint reloads on small synthetic graphs."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from gsf_chi.checkpoints import load_model
from gsf_chi.data.ecd import collate_ecd
from gsf_chi.ecd import ECDModel
from gsf_chi.tasks import axial_ecd, ranking


def samples(slots=7):
    rng = np.random.default_rng(7)
    out = []
    for base in range(3):
        n = 4
        sample = dict(
            id=f"{base}_a",
            base_id=base,
            node=rng.normal(size=(n, 52)).astype("float32"),
            pair=rng.random((n, n, 3), dtype=np.float32),
            unit_rel=rng.random((1, n, 26), dtype=np.float32),
            unit_desc=rng.random((1, 342), dtype=np.float32),
            phase_init=rng.random((1, n), dtype=np.float32),
            chi=np.ones(1, dtype=np.float32),
            rho=np.ones(1, dtype=np.float32),
            local_chi=np.ones((n, 1), dtype=np.float32),
            label=1,
            peak_num=2,
            peak_position=np.array([3, 8] + [-1] * (slots - 2)),
            peak_height=np.array([0, 1] + [0] * (slots - 2)),
        )
        mirror = copy.deepcopy(sample)
        mirror.update(
            id=f"{base}_b",
            chi=-sample["chi"],
            local_chi=-sample["local_chi"],
            label=0,
            peak_height=1 - sample["peak_height"],
        )
        out.extend([sample, mirror])
    return out


@pytest.mark.parametrize(
    "task,slots", [("axial_ecd", 7), ("central_ecd", 9), ("axial_rotation", 7)]
)
def test_one_epoch_and_checkpoint_reload(task, slots, tmp_path, monkeypatch):
    module = importlib.import_module("gsf_chi.tasks." + task)
    monkeypatch.setattr(
        "sys.argv",
        [
            "test",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--d-model",
            "24",
            "--n-layers",
            "1",
            "--device",
            "cpu",
            "--save-checkpoint",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    args = module.parse_args()
    data = samples(slots)
    split = {"train_index": [0, 1], "val_index": [2, 3], "test_index": [4, 5]}
    trainer = axial_ecd.train_one if task == "central_ecd" else module.train_one
    result = trainer("gsf", 7, data, split, args)
    assert result["best_epoch"] == 1
    loaded = load_model(Path(result["checkpoint_path"]), task)
    batch = collate_ecd(data[:2])
    prediction = loaded(batch)
    if task.endswith("ecd"):
        assert prediction[1].shape == (2, slots, 20)
        torch.testing.assert_close(prediction[0][0], prediction[0][1], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            prediction[2][0], prediction[2][1].flip(-1), atol=2e-6, rtol=2e-6
        )
    else:
        assert prediction.shape == (2, 2)


def test_central_and_axial_models_keep_separate_slot_counts():
    torch.manual_seed(8)
    axial = ECDModel("gsf", 24, 4, 1, 0.0, 5.0, n_peak_slots=7).eval()
    central = ECDModel("gsf", 24, 4, 1, 0.0, 5.0, n_peak_slots=9).eval()
    batch = collate_ecd(samples()[:2])
    first = axial(batch)
    assert central(batch)[1].shape[1] == 9
    second = axial(batch)
    for x, y in zip(first[:3], second[:3]):
        torch.testing.assert_close(x, y, atol=0, rtol=0)


def test_ranking_margin_and_ties():
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.5, 0.5])
    ordered = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    wrong = ordered.reshape(-1, 2).flip(-1).reshape(-1)
    _, _, good = ranking.objective(ordered, labels, 0.3, 1.0, 1.0)
    _, _, bad = ranking.objective(wrong, labels, 0.3, 1.0, 1.0)
    assert good == 0 and bad > good
    with pytest.raises(ValueError, match="complete pairs"):
        ranking.objective(ordered[:3], labels[:3], 0.3, 1.0, 1.0)


@pytest.mark.parametrize(
    "config", sorted((Path(__file__).parents[1] / "configs").rglob("*.json")), ids=lambda p: p.stem
)
def test_configs_parse_and_cli_overrides(config, monkeypatch):
    payload = json.loads(config.read_text())
    module = importlib.import_module("gsf_chi.tasks." + payload["task"])
    monkeypatch.setattr("sys.argv", ["test", "--config", str(config), "--device", "cpu"])
    args = module.parse_args()
    assert args.device == "cpu"
    assert args.seeds == payload["args"]["seeds"]


@pytest.mark.parametrize("task", ["rs", "ranking"])
def test_configuration_and_ranking_training(task, tmp_path, monkeypatch):
    from gsf_chi.batching import collate

    module = importlib.import_module("gsf_chi.tasks." + task)
    extra = (
        [
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--ema-decay",
            "0.9",
            "--mirror-loss-weight",
            "0.5",
            "--mirror-consistency-weight",
            "0.3",
        ]
        if task == "ranking"
        else []
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "test",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--d-model",
            "24",
            "--n-heads",
            "4",
            "--n-chiral-heads",
            "2",
            "--n-layers",
            "1",
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "result.json"),
        ]
        + extra,
    )
    args = module.parse_args()
    data = samples()
    loaders = {}
    for name, start in [("train", 0), ("validation", 2), ("test", 4)]:
        batch = collate(data[start : start + 2])
        batch["stereo_id"] = batch["id"]
        batch["base_key"] = [str(x) for x in batch["base_id"].tolist()]
        if task == "ranking":
            batch["label"] = batch["label"].float()
        loaders[name] = [batch]

    class Sampler:
        def set_epoch(self, epoch):
            pass

    monkeypatch.setattr(module, "_make_loaders", lambda *a, **k: (loaders, Sampler()))
    result = module.train_one("gsf", 7, args)
    assert result["best"]["epoch"] == 1
    assert np.isfinite(result["test"]["loss"])
    if task == "ranking":
        restored = load_model(Path(result["checkpoint"]), "ranking")
        assert torch.isfinite(restored(loaders["test"][0])).all()
