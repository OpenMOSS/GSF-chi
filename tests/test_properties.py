import copy
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from rdkit import Chem

import gsf_chi.attention as gsf_attention
import gsf_chi.batching as gsf_batching
import gsf_chi.data.axial as gsf_data_axial
import gsf_chi.data.central_ecd as gsf_data_central_ecd
import gsf_chi.data.ecd as gsf_data_ecd
import gsf_chi.ecd as gsf_ecd
import gsf_chi.ecd_metrics as gsf_ecd_metrics
import gsf_chi.model as gsf_model
import gsf_chi.paths as gsf_paths
import gsf_chi.tasks.ranking as gsf_tasks_ranking
import gsf_chi.training as gsf_training

EXPERIMENTS = Path(__file__).resolve().parent


def rodrigues(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    axis = torch.nn.functional.normalize(axis, dim=-1)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1).reshape(
        axis.shape[:-1] + (3, 3)
    )
    eye = torch.eye(3, dtype=axis.dtype, device=axis.device)
    return (
        eye
        + torch.sin(theta)[..., None, None] * skew
        + (1.0 - torch.cos(theta))[..., None, None] * (skew @ skew)
    )


def test_rodrigues_inverse_and_pair_reversal():
    torch.manual_seed(0)
    axis = torch.randn(8, 3, dtype=torch.float64)
    theta = torch.randn(8, dtype=torch.float64)
    u = rodrigues(axis, theta)
    u_chi_flip = rodrigues(axis, -theta)
    u_pair_reverse = rodrigues(axis, -theta)
    assert torch.allclose(u_chi_flip, u.transpose(-1, -2), atol=1e-12)
    assert torch.allclose(u_pair_reverse, u.transpose(-1, -2), atol=1e-12)
    assert torch.allclose(u @ u_chi_flip, torch.eye(3, dtype=u.dtype), atol=1e-12)


def test_fixed_axis_rotations_commute():
    """Documents that the submitted fixed-axis parameterization is an Abelian subgroup."""
    axis = torch.tensor([0.3, -0.7, 0.2], dtype=torch.float64)
    u = rodrigues(axis, torch.tensor(0.7, dtype=torch.float64))
    v = rodrigues(axis, torch.tensor(-1.3, dtype=torch.float64))
    assert torch.allclose(u @ v, v @ u, atol=1e-12)


def test_model_ema_is_single_state_dict_and_copies_buffers():
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4), torch.nn.Linear(4, 1)
    )
    ema = gsf_training.ModelEMA(model, decay=0.9)
    before = {name: parameter.detach().clone() for name, parameter in ema.model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter))
        model[1].running_mean.fill_(3.0)
        model[1].num_batches_tracked.fill_(11)
    ema.update()
    for name, parameter in ema.model.named_parameters():
        expected = before[name].lerp(dict(model.named_parameters())[name], 0.1)
        assert torch.allclose(parameter, expected)
    assert torch.equal(ema.model[1].running_mean, model[1].running_mean)
    assert torch.equal(ema.model[1].num_batches_tracked, model[1].num_batches_tracked)
    assert set(ema.model.state_dict()) == set(model.state_dict())


def test_chi_flip_counterfactual_preserves_even_inputs_and_swaps_targets():
    batch = {
        "chi": torch.tensor([[1.0], [-1.0], [-1.0], [1.0]]),
        "local_chi": torch.tensor([[[1.0]], [[-1.0]], [[-1.0]], [[1.0]]]),
        "pair": torch.randn(4, 3, 3, 4),
        "label": torch.tensor([1.2, 0.7, -0.4, 0.3]),
    }
    mirrored = gsf_tasks_ranking._mirror_chirality(batch)
    assert torch.equal(mirrored["chi"], -batch["chi"])
    assert torch.equal(mirrored["local_chi"], -batch["local_chi"])
    assert mirrored["pair"] is batch["pair"]
    assert torch.equal(
        gsf_tasks_ranking._swap_pair_targets(batch["label"]), torch.tensor([0.7, 1.2, 0.3, -0.4])
    )
    assert torch.equal(batch["chi"], torch.tensor([[1.0], [-1.0], [-1.0], [1.0]]))
    with pytest.raises(ValueError, match="complete pairs"):
        gsf_tasks_ranking._swap_pair_targets(torch.ones(3))


def test_pairwise_stereo_unit_dropout_shares_masks_only_within_pairs():
    rho = torch.ones(8, 4)
    torch.manual_seed(23)
    keep = gsf_attention._stereo_unit_dropout_keep(rho, 0.5, pairwise=True)
    assert torch.equal(keep[0], keep[1])
    assert torch.equal(keep[2], keep[3])
    assert torch.equal(keep[4], keep[5])
    assert torch.equal(keep[6], keep[7])
    assert set(keep.unique().tolist()) == {0.0, 1.0}


def test_physical_unit_dropout_preserves_chi_angle_and_scales_amplitude_once():
    rho = torch.ones(8, 4)
    chi = torch.ones(8, 4)
    torch.manual_seed(31)
    dropped_rho, dropped_chi = gsf_attention._apply_stereo_unit_dropout(
        rho, chi, 0.5, pairwise=True, preserve_chi_magnitude=True
    )
    assert torch.equal(dropped_chi[0], dropped_chi[1])
    assert set(dropped_chi.unique().tolist()) == {0.0, 1.0}
    assert set(dropped_rho.unique().tolist()) == {0.0, 2.0}
    assert torch.equal(dropped_rho, 2.0 * dropped_chi)


@pytest.mark.data
def test_endpoint_swap_invariance_of_even_field_and_chi():
    with open(gsf_paths.ROOT / "data" / "hct_ecd_axial.pkl", "rb") as handle:
        raw = pickle.load(handle)
    with open(gsf_paths.ROOT / "data" / "hct_ecd_axial_res.pkl", "rb") as handle:
        results = pickle.load(handle)
    labels = pd.read_excel(gsf_paths.ROOT / "data" / "axial_650.xlsx")
    for row, result in zip(raw, results):
        base_id, _ = gsf_data_axial._id_parts(row["id"])
        units = [tuple(x) for x in eval(labels.iloc[base_id]["label"])]
        if len(units) == 1 and len(units[0]) == 2:
            break
    mol = row["rdkit_mol"]
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
    unit = units[0]
    chi, abs_sin, cos_phi = gsf_data_axial._unit_chi_and_shape(mol, unit, result, ranks)
    chi_rev, abs_sin_rev, cos_phi_rev = gsf_data_axial._unit_chi_and_shape(
        mol, unit[::-1], result, ranks
    )
    graph_dist = np.asarray(Chem.GetDistanceMatrix(mol), dtype=np.float32)
    rel = gsf_data_axial._unit_relations(mol, unit, graph_dist, abs_sin, cos_phi)
    rel_rev = gsf_data_axial._unit_relations(mol, unit[::-1], graph_dist, abs_sin_rev, cos_phi_rev)
    assert chi == chi_rev
    assert np.allclose(rel, rel_rev, atol=1e-06)
    assert np.allclose(
        gsf_data_axial._phase_initialization(rel),
        gsf_data_axial._phase_initialization(rel_rev),
        atol=1e-06,
    )


@pytest.mark.data
def test_atom_permutation_and_nonchiral_reduction():
    samples, _ = gsf_data_axial.load_acmp(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_batching.collate([sample])
    torch.manual_seed(1)
    gsf = gsf_model.MolecularModel("gsf", d_model=24, n_heads=4, n_layers=1, dropout=0)
    gsf.eval()
    with torch.no_grad():
        reference = gsf(batch)
    n = len(sample["node"])
    perm = torch.randperm(n)
    permuted = dict(batch)
    for key in ["node", "local_chi", "node_mask"]:
        permuted[key] = batch[key][:, perm]
    permuted["pair"] = batch["pair"][:, perm][:, :, perm]
    permuted["unit_rel"] = batch["unit_rel"][:, :, perm]
    permuted["phase_init"] = batch["phase_init"][:, :, perm]
    with torch.no_grad():
        actual = gsf(permuted)
    assert torch.allclose(reference, actual, atol=2e-06)
    base = gsf_model.MolecularModel("base", d_model=24, n_heads=4, n_layers=1, dropout=0)
    base.load_state_dict(gsf.state_dict())
    base.eval()
    achiral = dict(batch)
    achiral["unit_mask"] = torch.zeros_like(batch["unit_mask"])
    achiral["chi"] = torch.zeros_like(batch["chi"])
    achiral["rho"] = torch.zeros_like(batch["rho"])
    with torch.no_grad():
        assert torch.allclose(gsf(achiral), base(achiral), atol=1e-07)


@pytest.mark.data
def test_zero_initialized_local_and_token_residuals_equal_gsf():
    samples, _ = gsf_data_axial.load_acmp(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_batching.collate([sample])
    torch.manual_seed(17)
    reference = gsf_model.MolecularModel("gsf", d_model=24, n_heads=4, n_layers=1, dropout=0.0)
    reference.eval()
    with torch.no_grad():
        expected = reference(batch)
    for mode in ("gsf_local", "gsf_token"):
        combined = gsf_model.MolecularModel(mode, d_model=24, n_heads=4, n_layers=1, dropout=0.0)
        combined.load_state_dict(reference.state_dict())
        torch.nn.init.zeros_(combined.chi_embed.weight)
        combined.eval()
        with torch.no_grad():
            actual = combined(batch)
        assert torch.equal(expected, actual)


@pytest.mark.data
def test_acmp_mirror_pairs_have_equal_even_inputs_and_inverse_chi():
    samples, split = gsf_data_axial.load_acmp(gsf_paths.ROOT / "data")
    audit = gsf_data_axial.verify_dataset(samples, split)
    assert audit["all_chi_vectors_invert"]
    assert audit["all_parity_even_inputs_match"]
    assert all((value == 0 for value in audit["split_pair_overlap"].values()))


@pytest.mark.data
def test_translation_and_proper_rotation_do_not_change_preprocessing():
    with open(gsf_paths.ROOT / "data" / "hct_ecd_axial.pkl", "rb") as handle:
        raw = pickle.load(handle)
    with open(gsf_paths.ROOT / "data" / "hct_ecd_axial_res.pkl", "rb") as handle:
        results = pickle.load(handle)
    labels = pd.read_excel(gsf_paths.ROOT / "data" / "axial_650.xlsx")
    optical = pd.read_csv(gsf_paths.ROOT / "data" / "optical_rotation_589nm.csv")
    optical_by_id = dict(zip(optical["id"].astype(int), optical["OR_589nm"].astype(float)))
    row = raw[2]
    base_id, _ = gsf_data_axial._id_parts(row["id"])
    reference = gsf_data_axial.build_sample(row, results[2], labels.iloc[base_id], optical_by_id)
    transformed_row = dict(row)
    mol = copy.deepcopy(row["rdkit_mol"])
    xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float64)
    angle = 0.73
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    xyz = xyz @ rotation.T + np.array([3.2, -1.1, 0.7])
    conf = mol.GetConformer()
    for idx, point in enumerate(xyz):
        conf.SetAtomPosition(idx, point.tolist())
    transformed_row["rdkit_mol"] = mol
    transformed = gsf_data_axial.build_sample(
        transformed_row, results[2], labels.iloc[base_id], optical_by_id
    )
    for key in ["node", "pair", "unit_rel", "unit_desc", "phase_init", "chi", "rho", "local_chi"]:
        assert np.allclose(reference[key], transformed[key], atol=2e-06), key


@pytest.mark.data
def test_ecd_parity_readout_is_exactly_even_odd():
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_data_ecd.collate_ecd([sample])
    mirrored = dict(batch)
    mirrored["chi"] = -batch["chi"]
    mirrored["local_chi"] = -batch["local_chi"]
    for mode in ["token", "local", "gsf"]:
        torch.manual_seed(7)
        model = gsf_ecd.ECDModel(
            mode=mode,
            d_model=24,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
            chiral_scale_init=5.0,
            readout="parity_chain",
        )
        model.eval()
        with torch.no_grad():
            number, position, height, transition = model(batch)
            mirror_number, mirror_position, mirror_height, mirror_transition = model(mirrored)
        assert torch.allclose(number, mirror_number, atol=1e-06)
        assert torch.allclose(position, mirror_position, atol=1e-06)
        assert torch.allclose(height, mirror_height.flip(-1), atol=1e-06)
        assert torch.allclose(transition, mirror_transition, atol=1e-06)
        decoded = gsf_ecd_metrics.decode_height_sequence(height, transition)
        mirror_decoded = gsf_ecd_metrics.decode_height_sequence(mirror_height, mirror_transition)
        assert torch.equal(decoded, 1 - mirror_decoded)


@pytest.mark.data
def test_ecd_split_readout_is_exactly_even_odd():
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_data_ecd.collate_ecd([sample])
    mirrored = dict(batch)
    mirrored["chi"] = -batch["chi"]
    mirrored["local_chi"] = -batch["local_chi"]
    for mode in ["token", "local", "gsf"]:
        torch.manual_seed(41)
        model = gsf_ecd.ECDModel(
            mode=mode,
            d_model=24,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
            chiral_scale_init=5.0,
            readout="parity_split",
        )
        model.eval()
        with torch.no_grad():
            number, position, height, transition = model(batch)
            mirror_number, mirror_position, mirror_height, mirror_transition = model(mirrored)
        assert transition is None
        assert mirror_transition is None
        assert torch.allclose(number, mirror_number, atol=1e-06)
        assert torch.allclose(position, mirror_position, atol=1e-06)
        assert torch.allclose(height, mirror_height.flip(-1), atol=1e-06)


@pytest.mark.data
def test_gsf_mechanism_ablation_defaults_are_backward_compatible():
    """The new experimental switches must not change the submitted default."""
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_data_ecd.collate_ecd([sample])
    common = dict(
        mode="gsf",
        d_model=24,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        chiral_scale_init=5.0,
        readout="parity_split",
    )
    torch.manual_seed(53)
    implicit = gsf_ecd.ECDModel(**common).eval()
    torch.manual_seed(53)
    explicit = gsf_ecd.ECDModel(
        **common,
        gsf_scope="global",
        condition_unit_axis=True,
        condition_unit_frequency=True,
        learn_pair_gate=True,
        use_phase_initialization=True,
    ).eval()
    with torch.no_grad():
        implicit_outputs = implicit(batch)
        explicit_outputs = explicit(batch)
    for implicit_output, explicit_output in zip(implicit_outputs, explicit_outputs):
        if implicit_output is None:
            assert explicit_output is None
        else:
            assert torch.equal(implicit_output, explicit_output)


@pytest.mark.data
def test_gsf_mechanism_ablations_retain_exact_task_parity():
    """Every mechanism ablation preserves the even/odd ECD readout contract."""
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_data_ecd.collate_ecd([sample])
    mirrored = dict(batch)
    mirrored["chi"] = -batch["chi"]
    mirrored["local_chi"] = -batch["local_chi"]
    variants = [
        {"gsf_scope": "query_anchor"},
        {"gsf_scope": "incident_anchor"},
        {"condition_unit_axis": False},
        {"condition_unit_frequency": False},
        {"learn_pair_gate": False},
        {"use_phase_initialization": False},
    ]
    reference_parameter_count = None
    for offset, variant in enumerate(variants):
        torch.manual_seed(59 + offset)
        model = gsf_ecd.ECDModel(
            mode="gsf",
            d_model=24,
            n_heads=4,
            n_layers=1,
            dropout=0.0,
            chiral_scale_init=5.0,
            readout="parity_split",
            **variant,
        ).eval()
        parameter_count = sum((parameter.numel() for parameter in model.parameters()))
        if reference_parameter_count is None:
            reference_parameter_count = parameter_count
        assert parameter_count == reference_parameter_count
        with torch.no_grad():
            number, position, height, transition = model(batch)
            mirror_number, mirror_position, mirror_height, mirror_transition = model(mirrored)
        assert transition is None
        assert mirror_transition is None
        assert torch.allclose(number, mirror_number, atol=1e-06)
        assert torch.allclose(position, mirror_position, atol=1e-06)
        assert torch.allclose(height, mirror_height.flip(-1), atol=1e-06)


@pytest.mark.data
def test_captured_unit_corrections_sum_to_layer_correction():
    samples, _ = gsf_data_axial.load_acmp(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) > 1))
    batch = gsf_batching.collate([sample])
    torch.manual_seed(67)
    model = gsf_model.MolecularModel("gsf", d_model=24, n_heads=4, n_layers=1, dropout=0.0).eval()
    layer = model.layers[0]
    layer.capture_diagnostics = True
    with torch.no_grad():
        model(batch)
    assert layer.last_phase is not None
    assert layer.last_pair_gate is not None
    assert layer.last_unit_correction is not None
    assert layer.last_chiral_correction is not None
    summed = layer.last_unit_correction.sum(dim=1)
    assert torch.allclose(summed, layer.last_chiral_correction, atol=2e-06, rtol=1e-06)


@pytest.mark.data
def test_gsf_scope_masks_form_the_claimed_support_hierarchy():
    samples, _ = gsf_data_axial.load_acmp(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_batching.collate([sample])
    common = dict(mode="gsf", d_model=24, n_heads=4, n_layers=1, dropout=0.0)
    torch.manual_seed(71)
    global_model = gsf_model.MolecularModel(**common, gsf_scope="global").eval()
    state = global_model.state_dict()
    models = {"global": global_model}
    for scope in ("query_anchor", "incident_anchor"):
        model = gsf_model.MolecularModel(**common, gsf_scope=scope).eval()
        model.load_state_dict(state)
        models[scope] = model
    gates = {}
    for name, model in models.items():
        model.layers[0].capture_diagnostics = True
        with torch.no_grad():
            model(batch)
        gates[name] = model.layers[0].last_pair_gate[0, 0]
    anchor = batch["unit_rel"][0, 0, :, -2].bool()
    remote = ~anchor
    assert torch.count_nonzero(gates["query_anchor"][remote, :]) == 0
    assert torch.count_nonzero(gates["incident_anchor"][remote][:, remote]) == 0
    assert torch.all(gates["global"] > 0)
    assert torch.all(gates["query_anchor"].bool() <= gates["incident_anchor"].bool())
    assert torch.all(gates["incident_anchor"].bool() <= gates["global"].bool())


@pytest.mark.data
def test_ecd_split_symbol_loss_does_not_update_position_head():
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_data_ecd.collate_ecd([sample])
    torch.manual_seed(43)
    model = gsf_ecd.ECDModel(
        mode="gsf",
        d_model=24,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        chiral_scale_init=5.0,
        readout="parity_split",
    )
    model.train()
    _, _, height, _ = model(batch)
    height[..., 1].sum().backward()
    position_gradients = [parameter.grad for parameter in model.position.parameters()]
    symbol_context_gradients = [parameter.grad for parameter in model.symbol_position.parameters()]
    assert all((gradient is None for gradient in position_gradients))
    assert any(
        (
            gradient is not None and torch.count_nonzero(gradient).item() > 0
            for gradient in symbol_context_gradients
        )
    )


@pytest.mark.data
def test_ecd_split_chain_is_exactly_even_odd_and_gradient_isolated():
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_data_ecd.collate_ecd([sample])
    mirrored = dict(batch)
    mirrored["chi"] = -batch["chi"]
    mirrored["local_chi"] = -batch["local_chi"]
    torch.manual_seed(47)
    model = gsf_ecd.ECDModel(
        mode="gsf",
        d_model=24,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        chiral_scale_init=5.0,
        readout="parity_split_chain",
    )
    model.eval()
    with torch.no_grad():
        number, position, height, transition = model(batch)
        mirror_number, mirror_position, mirror_height, mirror_transition = model(mirrored)
    assert transition is not None
    assert torch.allclose(number, mirror_number, atol=1e-06)
    assert torch.allclose(position, mirror_position, atol=1e-06)
    assert torch.allclose(height, mirror_height.flip(-1), atol=1e-06)
    assert torch.allclose(transition, mirror_transition, atol=1e-06)
    decoded = gsf_ecd_metrics.decode_height_sequence(height, transition)
    mirror_decoded = gsf_ecd_metrics.decode_height_sequence(mirror_height, mirror_transition)
    assert torch.equal(decoded, 1 - mirror_decoded)
    model.train()
    model.zero_grad(set_to_none=True)
    _, _, height, transition = model(batch)
    (height[..., 1].sum() + transition[..., 1].sum()).backward()
    assert all((parameter.grad is None for parameter in model.position.parameters()))
    assert any(
        (
            parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
            for parameter in model.symbol_position.parameters()
        )
    )


def test_monotonic_position_decoder_finds_best_strict_sequence():
    logits = torch.full((2, gsf_ecd.N_PEAK_SLOTS, gsf_ecd.N_POSITION_CLASSES), -1000.0)
    logits[0, 0, 5] = 100.0
    logits[0, 0, 1] = 80.0
    logits[0, 1, 4] = 100.0
    logits[0, 1, 6] = 50.0
    logits[0, 2, 3] = 100.0
    logits[0, 2, 7] = 20.0
    logits[1, 0, 9] = 3.0
    count = torch.tensor([3, 1])
    decoded = gsf_ecd_metrics.decode_monotonic_positions(logits, count)
    assert decoded[0, :3].tolist() == [1, 4, 7]
    assert decoded[1, 0].item() == 9
    assert torch.all(decoded[0, 1:3] > decoded[0, :2])


def test_posterior_mean_position_decoder_is_squared_error_decision():
    logits = torch.full((1, 2, 5), -20.0)
    logits[0, 0, 0] = math.log(0.55)
    logits[0, 0, 4] = math.log(0.45)
    logits[0, 1, 3] = 20.0
    decoded = gsf_ecd_metrics.decode_posterior_mean_positions(logits)
    assert decoded.tolist() == [[2, 3]]


def test_monotonic_position_loss_penalizes_reversed_expectations():
    slots = gsf_ecd.N_PEAK_SLOTS
    classes = gsf_ecd.N_POSITION_CLASSES
    number = torch.zeros(1, slots)
    number[0, 3] = 10.0
    height = torch.zeros(1, slots, 2)
    target_position = torch.full((1, slots), -1, dtype=torch.long)
    target_height = torch.full((1, slots), -1, dtype=torch.long)
    target_position[0, :3] = torch.tensor([1, 4, 7])
    target_height[0, :3] = torch.tensor([0, 1, 0])
    batch = {
        "peak_num": torch.tensor([3]),
        "peak_position": target_position,
        "peak_height": target_height,
    }
    ordered = torch.full((1, slots, classes), -20.0)
    reversed_position = torch.full((1, slots, classes), -20.0)
    for slot_index, position_index in enumerate([1, 4, 7]):
        ordered[0, slot_index, position_index] = 20.0
    for slot_index, position_index in enumerate([7, 4, 1]):
        reversed_position[0, slot_index, position_index] = 20.0
    _, ordered_terms = gsf_ecd_metrics.loss_terms((number, ordered, height, None), batch)
    _, reversed_terms = gsf_ecd_metrics.loss_terms((number, reversed_position, height, None), batch)
    assert ordered_terms["position_monotonic"] < 1e-06
    assert reversed_terms["position_monotonic"] > 1.0


def test_molecule_position_loss_matches_equal_molecule_weighting():
    slots = gsf_ecd.N_PEAK_SLOTS
    classes = gsf_ecd.N_POSITION_CLASSES
    number = torch.zeros(2, slots)
    height = torch.zeros(2, slots, 2)
    position = torch.zeros(2, slots, classes, requires_grad=True)
    target_position = torch.full((2, slots), -1, dtype=torch.long)
    target_height = torch.full((2, slots), -1, dtype=torch.long)
    target_position[0, 0] = 2
    target_position[1, :3] = torch.tensor([3, 7, 11])
    target_height[0, 0] = 0
    target_height[1, :3] = torch.tensor([0, 1, 0])
    batch = {
        "peak_num": torch.tensor([1, 3]),
        "peak_position": target_position,
        "peak_height": target_height,
    }
    loss, terms = gsf_ecd_metrics.loss_terms(
        (number, position, height, None),
        batch,
        position_loss_reduction="molecule",
        position_risk_reduction="molecule_rmse",
        position_soft_label_sigma=0.75,
        position_risk_weight=0.15,
    )
    assert math.isfinite(terms["position"])
    assert math.isfinite(terms["position_risk"])
    loss.backward()
    assert position.grad is not None
    assert torch.isfinite(position.grad).all()


def test_official_position_mask_uses_frozen_predicted_count_only_for_position():
    slots = gsf_ecd.N_PEAK_SLOTS
    classes = gsf_ecd.N_POSITION_CLASSES
    number = torch.full((1, slots), -10.0)
    number[0, 1] = 10.0
    position = torch.full((1, slots, classes), -10.0, requires_grad=True)
    target_position = torch.full((1, slots), -1, dtype=torch.long)
    target_position[0, :3] = torch.tensor([2, 7, 11])
    position.data[0, 0, 2] = 10.0
    position.data[0, 1, 18] = 10.0
    position.data[0, 2, 18] = 10.0
    height = torch.zeros(1, slots, 2, requires_grad=True)
    target_height = torch.full((1, slots), -1, dtype=torch.long)
    target_height[0, :3] = torch.tensor([0, 1, 0])
    batch = {
        "peak_num": torch.tensor([3]),
        "peak_position": target_position,
        "peak_height": target_height,
    }
    _, target_terms = gsf_ecd_metrics.loss_terms(
        (number, position, height, None), batch, position_mask_mode="target"
    )
    loss, official_terms = gsf_ecd_metrics.loss_terms(
        (number, position, height, None),
        batch,
        position_mask_mode="official_predicted",
        position_loss_reduction="molecule",
        position_ordinal_reduction="molecule_rmse",
        position_ordinal_weight=1.0,
    )
    assert official_terms["position"] < 1e-05
    assert target_terms["position"] > 5.0
    loss.backward()
    assert torch.count_nonzero(height.grad[0, :3]).item() > 0


def _sample_with_dense_spectrum_target() -> tuple[dict, dict]:
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    sample = copy.deepcopy(next((x for x in samples if len(x["chi"]) == 1)))
    spectrum_abs = np.zeros(gsf_ecd.N_POSITION_CLASSES, dtype=np.float32)
    count = min(int(sample["peak_num"]), gsf_ecd.N_PEAK_SLOTS)
    for index in sample["peak_position"][:count]:
        spectrum_abs[int(index)] = 1.0
    sample["spectrum_abs"] = spectrum_abs
    batch = gsf_data_ecd.collate_ecd([sample])
    return (sample, batch)


@pytest.mark.data
def test_backbone_atom_tokens_pool_to_legacy_encode():
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    batch = gsf_data_ecd.collate_ecd([samples[0], samples[1]])
    torch.manual_seed(59)
    backbone = gsf_model.MolecularModel(
        mode="gsf", d_model=24, n_heads=4, n_layers=1, dropout=0.0, chiral_scale_init=5.0
    ).eval()
    with torch.no_grad():
        nodes, node_mask = backbone.encode_nodes(batch)
        pooled = backbone.encode(batch)
    expected = (nodes * node_mask[..., None]).sum(dim=1) / node_mask.sum(
        dim=1, keepdim=True
    ).clamp_min(1)
    assert torch.allclose(pooled, expected, atol=1e-07)


def _add_mock_even_chemistry(sample: dict) -> dict:
    """Attach a small valid directed bond-angle graph for readout properties."""
    sample = copy.deepcopy(sample)
    n_atoms = len(sample["node"])
    undirected = [(index, index + 1) for index in range(n_atoms - 1)]
    directed = [edge for left, right in undirected for edge in ((left, right), (right, left))]
    bond_index = np.asarray(directed, dtype=np.int64)
    n_bonds = len(bond_index)
    angle_index = np.asarray(
        [(index, index + 1) for index in range(max(n_bonds - 1, 0))], dtype=np.int64
    ).reshape(-1, 2)
    sample.update(
        chem_bond_index=bond_index,
        chem_bond_type=np.asarray([1 + index % 4 for index in range(n_bonds)], dtype=np.int64),
        chem_bond_ring=np.zeros(n_bonds, dtype=np.int64),
        chem_bond_length=np.linspace(1.1, 1.6, n_bonds, dtype=np.float32),
        chem_angle_index=angle_index,
        chem_angle=np.linspace(0.4, 2.6, len(angle_index), dtype=np.float32),
        chem_global=np.asarray([0.3, 0.1, 0.25, 0.4, 0.2], dtype=np.float32),
    )
    return sample


@pytest.mark.data
def test_all_attention_heads_can_be_chiral():
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_data_ecd.collate_ecd([sample])
    mirrored = dict(batch)
    mirrored["chi"] = -batch["chi"]
    torch.manual_seed(31)
    model = gsf_ecd.ECDModel(
        mode="gsf",
        d_model=24,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        chiral_scale_init=5.0,
        readout="parity",
        n_chiral_heads=4,
    )
    assert model.backbone.layers[0].n_chiral_heads == 4
    model.eval()
    with torch.no_grad():
        number, position, height, _ = model(batch)
        mirror_number, mirror_position, mirror_height, _ = model(mirrored)
    assert torch.allclose(number, mirror_number, atol=1e-06)
    assert torch.allclose(position, mirror_position, atol=1e-06)
    assert torch.allclose(height, mirror_height.flip(-1), atol=1e-06)


@pytest.mark.data
def test_partial_so3_blocks_preserve_exact_ecd_parity():
    """A 32-wide head rotates 30 dimensions and leaves two ordinary dimensions."""
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if len(x["chi"]) == 1))
    batch = gsf_data_ecd.collate_ecd([sample])
    mirrored = dict(batch)
    mirrored["chi"] = -batch["chi"]
    torch.manual_seed(29)
    model = gsf_ecd.ECDModel(
        mode="gsf",
        d_model=256,
        n_heads=8,
        n_layers=1,
        dropout=0.0,
        chiral_scale_init=5.0,
        readout="parity",
    )
    assert model.backbone.layers[0].head_dim == 32
    assert model.backbone.layers[0].chiral_dim == 30
    model.eval()
    with torch.no_grad():
        number, position, height, _ = model(batch)
        mirror_number, mirror_position, mirror_height, _ = model(mirrored)
    assert torch.allclose(number, mirror_number, atol=1e-06)
    assert torch.allclose(position, mirror_position, atol=1e-06)
    assert torch.allclose(height, mirror_height.flip(-1), atol=1e-05)


@pytest.mark.data
def test_training_dropout_does_not_create_false_odd_signal():
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    batch = gsf_data_ecd.collate_ecd([samples[0], samples[1]])
    batch["chi"] = torch.zeros_like(batch["chi"])
    torch.manual_seed(19)
    model = gsf_ecd.ECDModel(
        mode="gsf",
        d_model=24,
        n_heads=4,
        n_layers=2,
        dropout=0.35,
        chiral_scale_init=5.0,
        readout="parity",
        stereo_unit_dropout=0.4,
    )
    model.train()
    _, z_odd = model.parity_representations(batch)
    assert torch.allclose(z_odd, torch.zeros_like(z_odd), atol=1e-07)


@pytest.mark.data
def test_stereo_unit_dropout_preserves_training_parity():
    samples, _ = gsf_data_ecd.load_ecd_samples(gsf_paths.ROOT / "data")
    sample = next((x for x in samples if np.any(x["chi"] != 0)))
    batch = gsf_data_ecd.collate_ecd([sample])
    mirrored = dict(batch)
    mirrored["chi"] = -batch["chi"]
    mirrored["local_chi"] = -batch["local_chi"]
    for mode in ["local", "gsf"]:
        torch.manual_seed(23)
        model = gsf_ecd.ECDModel(
            mode=mode,
            d_model=24,
            n_heads=4,
            n_layers=2,
            dropout=0.35,
            chiral_scale_init=5.0,
            readout="parity",
            stereo_unit_dropout=0.4,
        )
        model.train()
        rng_state = torch.get_rng_state()
        z_even, z_odd = model.parity_representations(batch)
        torch.set_rng_state(rng_state)
        mirror_even, mirror_odd = model.parity_representations(mirrored)
        assert torch.allclose(z_even, mirror_even, atol=1e-07)
        assert torch.allclose(z_odd, -mirror_odd, atol=1e-07)


@pytest.mark.data
def test_central_cip_branch_field_and_rs_inverse():
    records = np.load(
        gsf_paths.ROOT / "data" / "central_ecd_raw" / "ecd_column_charity_new_smiles.npy",
        allow_pickle=True,
    )
    first_entry = records[0]
    opposite_entry = next(
        (
            entry
            for entry in records
            if entry["hand_id"] == first_entry["hand_id"] and entry["id"] != first_entry["id"]
        )
    )
    first = gsf_data_central_ecd._graph_sample(first_entry, 1, "first")
    opposite = gsf_data_central_ecd._graph_sample(opposite_entry, 1, "opposite")
    assert np.array_equal(first["chi"], -opposite["chi"])
    relation = first["unit_rel"][0]
    phase = first["phase_init"][0]
    center = int(np.argmax(first["local_chi"][:, 0] != 0))
    assert phase[center] == 0.0
    branch_angles = np.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3, 0.0])
    branch_assignment = relation[:, 16:20]
    assigned = branch_assignment.sum(axis=1) > 0
    expected = branch_assignment[assigned] @ branch_angles + 1.2 * relation[assigned, 0]
    assert np.allclose(phase[assigned], expected, atol=1e-06)


@pytest.mark.data
def test_central_preprocessing_is_translation_and_proper_rotation_invariant():
    records = np.load(
        gsf_paths.ROOT / "data" / "central_ecd_raw" / "ecd_column_charity_new_smiles.npy",
        allow_pickle=True,
    )
    entry = dict(records[0])
    entry["info"] = dict(entry["info"])
    reference = gsf_data_central_ecd._graph_sample(entry, 1, "reference")
    xyz = np.asarray(entry["info"]["atom_pos"], dtype=np.float64)
    angle = 0.41
    rotation = np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ]
    )
    transformed_entry = dict(entry)
    transformed_entry["info"] = dict(entry["info"])
    transformed_entry["info"]["atom_pos"] = (xyz @ rotation.T + np.array([-2.0, 0.7, 4.1])).astype(
        np.float32
    )
    transformed = gsf_data_central_ecd._graph_sample(transformed_entry, 1, "transformed")
    for key in ["node", "pair", "unit_rel", "unit_desc", "phase_init", "chi", "rho", "local_chi"]:
        assert np.allclose(reference[key], transformed[key], atol=3e-06), key
