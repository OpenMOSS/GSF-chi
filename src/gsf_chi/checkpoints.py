"""Load a trained model with the arguments used to construct it."""

from __future__ import annotations

from pathlib import Path

import torch

from gsf_chi.ecd import ECDModel
from gsf_chi.model import MolecularModel, RankingModel


def load_model(path: Path, task: str, device="cpu", arguments: dict | None = None):
    payload = torch.load(path, map_location=device, weights_only=False)
    args = payload.get("args", arguments)
    if args is None:
        raise ValueError("This checkpoint predates saved arguments; pass its run JSON arguments.")
    mode = payload.get("mode", "gsf")
    support = args.get("gsf_support_mode", "full")
    kwargs = dict(
        mode=mode,
        gsf_support_mode="fixed" if support in {"random", "distance"} else support,
        gsf_support_fraction=args.get("gsf_support_fraction", 0.0),
        gsf_rotary_mode=args.get("gsf_rotary_mode", "residual"),
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        dropout=args["dropout"],
        chiral_scale_init=args["gsf_scale"],
        stereo_unit_dropout=args.get("stereo_unit_dropout", 0.0),
        n_chiral_heads=args.get("n_chiral_heads"),
        gsf_scope=args.get("gsf_scope", "global"),
        condition_unit_axis=not args.get("fixed_gsf_axis", False),
        condition_unit_frequency=not args.get("fixed_gsf_frequency", False),
        learn_pair_gate=not args.get("uniform_gsf_gate", False),
        use_phase_initialization=not args.get("zero_gsf_phase_init", False),
        share_gsf_axis_frequency=args.get("share_gsf_axis_frequency", False),
    )
    if task in {"axial_ecd", "central_ecd"}:
        kwargs.update(
            readout=args["readout"],
            number_readout=args.get("number_readout", "categorical"),
            position_decode=args.get("position_decode", "independent"),
            position_decode_temperature=args.get("position_decode_temperature", 1.0),
            n_peak_slots=7 if task == "axial_ecd" else 9,
        )
        model = ECDModel(**kwargs)
    elif task == "ranking":
        model = RankingModel(**kwargs)
    elif task in {"rs", "axial_rotation"}:
        model = MolecularModel(**kwargs)
    else:
        raise ValueError(f"Unsupported checkpoint task: {task}")
    state = dict(payload.get("state_dict", payload.get("model_state_dict", {})))
    if task in {"axial_ecd", "central_ecd"}:
        state.pop("count_position_prior", None)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()
