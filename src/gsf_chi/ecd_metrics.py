from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def decode_height_sequence(
    unary_logits: torch.Tensor, transition_logits: torch.Tensor | None
) -> torch.Tensor:
    """Viterbi decode binary peak signs; relation class 0=same, 1=flip."""
    if transition_logits is None:
        return unary_logits.argmax(dim=-1)
    _, n_slots, _ = unary_logits.shape
    relation = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=unary_logits.device)
    score = unary_logits[:, 0]
    backpointers = []
    for slot_index in range(1, n_slots):
        pair_score = transition_logits[:, slot_index - 1][:, relation]
        candidate = score[:, :, None] + unary_logits[:, slot_index, None, :] + pair_score
        score, previous = candidate.max(dim=1)
        backpointers.append(previous)
    state = score.argmax(dim=-1)
    path = [state]
    for previous in reversed(backpointers):
        state = previous.gather(1, state[:, None]).squeeze(1)
        path.append(state)
    return torch.stack(list(reversed(path)), dim=1)


def decode_monotonic_positions(
    position_logits: torch.Tensor, peak_count: torch.Tensor
) -> torch.Tensor:
    """Maximum-score strictly increasing Position sequence for each molecule.

    CMCDS stores peaks in strictly increasing wavelength-bin order.  Independent
    slot argmax can violate that known output support, so this Viterbi decoder
    finds the best valid sequence for the predicted number of peaks.  Slots
    beyond the predicted count retain their independent argmax because the
    official metric ignores them.
    """
    decoded = position_logits.argmax(dim=-1)
    _, n_slots, n_classes = position_logits.shape
    for batch_index in range(position_logits.shape[0]):
        count = min(int(peak_count[batch_index]), n_slots, n_classes)
        if count <= 1:
            continue
        score = position_logits[batch_index, 0]
        backpointers = []
        for slot_index in range(1, count):
            prefix_score, prefix_index = torch.cummax(score, dim=0)
            predecessor_score = torch.cat(
                [score.new_full((1,), -torch.inf), prefix_score[:-1]], dim=0
            )
            predecessor_index = torch.cat(
                [torch.full((1,), -1, dtype=torch.long, device=score.device), prefix_index[:-1]],
                dim=0,
            )
            score = position_logits[batch_index, slot_index] + predecessor_score
            backpointers.append(predecessor_index)
        state = int(score.argmax())
        path = [state]
        for predecessor in reversed(backpointers):
            state = int(predecessor[state])
            path.append(state)
        decoded[batch_index, :count] = torch.tensor(
            list(reversed(path)), dtype=torch.long, device=decoded.device
        )
    return decoded


def decode_posterior_mean_positions(
    position_logits: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """Rounded posterior mean, the Bayes decision for squared class error."""
    if temperature <= 0:
        raise ValueError("Position decode temperature must be positive")
    value = torch.arange(
        position_logits.shape[-1], device=position_logits.device, dtype=position_logits.dtype
    )
    probability = torch.softmax(position_logits / temperature, dim=-1)
    expectation = (probability * value).sum(dim=-1)
    return expectation.round().long().clamp_(0, position_logits.shape[-1] - 1)


def loss_terms(
    predictions: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None],
    batch: dict,
    number_loss_weight: float = 1.0,
    number_ordinal_weight: float = 0.0,
    number_risk_weight: float = 0.0,
    position_loss_weight: float = 1.0,
    position_ordinal_weight: float = 0.0,
    position_risk_weight: float = 0.0,
    position_monotonic_weight: float = 0.0,
    position_loss_reduction: str = "peak",
    position_risk_reduction: str = "peak_mse",
    position_ordinal_reduction: str = "peak_mse",
    position_soft_label_sigma: float = 0.0,
    position_mask_mode: str = "target",
) -> tuple[torch.Tensor, dict[str, float]]:
    number, position, height, transition = predictions
    gt_num = batch["peak_num"]
    if position_loss_reduction not in {"peak", "molecule"}:
        raise ValueError(f"Unknown Position loss reduction: {position_loss_reduction}")
    if position_risk_reduction not in {"peak_mse", "molecule_rmse"}:
        raise ValueError(f"Unknown Position risk reduction: {position_risk_reduction}")
    if position_ordinal_reduction not in {"peak_mse", "molecule_rmse"}:
        raise ValueError(f"Unknown Position ordinal reduction: {position_ordinal_reduction}")
    if position_mask_mode not in {"target", "official_predicted"}:
        raise ValueError(f"Unknown Position mask mode: {position_mask_mode}")
    if position_soft_label_sigma < 0:
        raise ValueError("Position soft-label sigma must be non-negative")
    slot = torch.arange(position.shape[1], device=gt_num.device)[None]
    target_valid = slot < gt_num[:, None]
    ordinal_number = number.shape[-1] == position.shape[1] - 1
    if ordinal_number:
        threshold = torch.arange(position.shape[1] - 1, device=number.device)[None, :]
        threshold_target = (gt_num[:, None] > threshold).to(number.dtype)
        loss_num = F.binary_cross_entropy_with_logits(number, threshold_target)
        expected_number = torch.sigmoid(number).sum(dim=-1)
    else:
        loss_num = F.cross_entropy(number, gt_num)
        number_values = torch.arange(number.shape[-1], device=number.device, dtype=number.dtype)
        expected_number = (torch.softmax(number, dim=-1) * number_values).sum(dim=-1)
    loss_num_ordinal = F.mse_loss(expected_number, gt_num.to(number.dtype))
    if ordinal_number:
        loss_num_risk = loss_num_ordinal
    else:
        number_distance_sq = (number_values[None, :] - gt_num[:, None].to(number.dtype)).square()
        loss_num_risk = (torch.softmax(number, dim=-1) * number_distance_sq).sum(dim=-1).mean()
    if position_mask_mode == "official_predicted":
        if ordinal_number:
            predicted_number = (number.detach() > 0.0).sum(dim=-1)
        else:
            predicted_number = number.detach().argmax(dim=-1)
        position_count = torch.minimum(gt_num, predicted_number)
        valid = slot < position_count[:, None]
    else:
        position_count = gt_num
        valid = target_valid
    position_values_all = torch.arange(
        position.shape[-1], device=position.device, dtype=position.dtype
    )
    position_probability_all = torch.softmax(position, dim=-1)
    expected_position_all = (position_probability_all * position_values_all).sum(dim=-1)
    if target_valid.any():
        loss_height = F.cross_entropy(height[target_valid], batch["peak_height"][target_valid])
    else:
        loss_height = height.sum() * 0.0
    if valid.any():
        position_values = position_values_all
        target_all = batch["peak_position"].clamp_min(0)
        if position_soft_label_sigma > 0:
            soft_target = torch.exp(
                -0.5
                * (
                    (position_values[None, None] - target_all[..., None].to(position.dtype))
                    / position_soft_label_sigma
                ).square()
            )
            soft_target = soft_target / soft_target.sum(dim=-1, keepdim=True).clamp_min(1e-08)
            slot_position_loss = -(soft_target * torch.log_softmax(position, dim=-1)).sum(dim=-1)
        else:
            slot_position_loss = F.cross_entropy(
                position.reshape(-1, position.shape[-1]), target_all.reshape(-1), reduction="none"
            ).reshape_as(target_all)
        molecule_peak_count = valid.sum(dim=-1).clamp_min(1).to(position.dtype)
        nonempty = position_count > 0
        if position_loss_reduction == "molecule":
            loss_position = ((slot_position_loss * valid).sum(dim=-1) / molecule_peak_count)[
                nonempty
            ].mean()
        else:
            loss_position = slot_position_loss[valid].mean()
        valid_target = batch["peak_position"][valid]
        position_probability = position_probability_all[valid]
        expected_position = (position_probability * position_values[None, :]).sum(dim=-1)
        if position_ordinal_reduction == "molecule_rmse":
            posterior_mean_error_sq = (
                expected_position_all - target_all.to(position.dtype)
            ).square()
            loss_position_ordinal = torch.sqrt(
                (posterior_mean_error_sq * valid).sum(dim=-1) / molecule_peak_count + 1e-08
            )[nonempty].mean()
        else:
            loss_position_ordinal = F.mse_loss(expected_position, valid_target.to(position.dtype))
        position_distance_sq_all = (
            position_values[None, None, :] - target_all[..., None].to(position.dtype)
        ).square()
        slot_position_risk = (position_probability_all * position_distance_sq_all).sum(dim=-1)
        if position_risk_reduction == "molecule_rmse":
            loss_position_risk = torch.sqrt(
                (slot_position_risk * valid).sum(dim=-1) / molecule_peak_count + 1e-08
            )[nonempty].mean()
        else:
            loss_position_risk = slot_position_risk[valid].mean()
    else:
        loss_position = position.sum() * 0.0
        loss_position_ordinal = position.sum() * 0.0
        loss_position_risk = position.sum() * 0.0
    valid_position_pair = slot[:, :-1] + 1 < position_count[:, None]
    if valid_position_pair.any():
        monotonic_violation = F.relu(
            1.0 + expected_position_all[:, :-1] - expected_position_all[:, 1:]
        )
        loss_position_monotonic = monotonic_violation[valid_position_pair].square().mean()
    else:
        loss_position_monotonic = position.sum() * 0.0
    loss_transition = height.sum() * 0.0
    if transition is not None:
        valid_transition = slot[:, :-1] + 1 < gt_num[:, None]
        transition_target = (batch["peak_height"][:, 1:] != batch["peak_height"][:, :-1]).long()
        if valid_transition.any():
            loss_transition = F.cross_entropy(
                transition[valid_transition], transition_target[valid_transition]
            )
    total = (
        number_loss_weight * loss_num
        + number_ordinal_weight * loss_num_ordinal
        + number_risk_weight * loss_num_risk
        + position_loss_weight * loss_position
        + position_ordinal_weight * loss_position_ordinal
        + position_risk_weight * loss_position_risk
        + position_monotonic_weight * loss_position_monotonic
        + 2.0 * loss_height
        + 0.5 * loss_transition
    )
    return (
        total,
        {
            "number": float(loss_num.detach()),
            "number_ordinal": float(loss_num_ordinal.detach()),
            "number_risk": float(loss_num_risk.detach()),
            "position": float(loss_position.detach()),
            "position_ordinal": float(loss_position_ordinal.detach()),
            "position_risk": float(loss_position_risk.detach()),
            "position_monotonic": float(loss_position_monotonic.detach()),
            "height": float(loss_height.detach()),
            "transition": float(loss_transition.detach()),
        },
    )


@dataclass
class ECDMetrics:
    loss: float
    number_rmse: float
    position_rmse: float
    height_accuracy: float
    transition_accuracy: float
    pair_height_complement_rate: float


def validation_selection_score(metrics: ECDMetrics, args: argparse.Namespace) -> float:
    """Validation-only checkpoint score under the declared experiment contract."""
    objective = getattr(args, "selection_objective", "composite")
    if objective == "composite":
        return (
            metrics.height_accuracy
            - getattr(args, "selection_number_weight", 0.0) * metrics.number_rmse
            - getattr(args, "selection_position_weight", 0.0) * metrics.position_rmse
        )
    if objective == "position":
        return -metrics.position_rmse
    if objective != "scaled_minimum":
        raise ValueError(f"Unknown validation selection objective: {objective}")
    scales = (
        args.selection_symbol_scale,
        args.selection_number_scale,
        args.selection_position_scale,
    )
    if any((scale <= 0 for scale in scales)):
        raise ValueError("Validation margin scales must be positive")
    margins = (
        (metrics.height_accuracy - args.selection_symbol_target) / args.selection_symbol_scale,
        (args.selection_number_target - metrics.number_rmse) / args.selection_number_scale,
        (args.selection_position_target - metrics.position_rmse) / args.selection_position_scale,
    )
    return min(margins)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> ECDMetrics:
    model.eval()
    total_loss = 0.0
    n_molecules = 0
    number_sq_error = 0.0
    molecule_position_rmse = []
    height_correct = 0
    height_total = 0
    transition_correct = 0
    transition_total = 0
    pair_height_predictions: dict[int, list[list[int]]] = {}
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        predictions = model(batch)
        loss, _ = loss_terms(predictions, batch)
        number, position, height, transition = predictions
        bsz = len(batch["peak_num"])
        total_loss += float(loss) * bsz
        n_molecules += bsz
        if number.shape[-1] == position.shape[1] - 1:
            pred_num = (number > 0.0).sum(dim=-1)
        else:
            pred_num = number.argmax(dim=-1)
        position_decode = getattr(model, "position_decode", "independent")
        if position_decode == "monotonic":
            pred_position = decode_monotonic_positions(position, pred_num)
        elif position_decode == "posterior_mean":
            pred_position = decode_posterior_mean_positions(
                position, getattr(model, "position_decode_temperature", 1.0)
            )
        else:
            pred_position = position.argmax(dim=-1)
        decode_transition = (
            transition if model.readout in {"parity_chain", "parity_split_chain"} else None
        )
        pred_height = decode_height_sequence(height, decode_transition)
        number_sq_error += ((pred_num - batch["peak_num"]).float() ** 2).sum().item()
        for idx in range(bsz):
            gt_n = int(batch["peak_num"][idx])
            pred_n = int(pred_num[idx])
            compare_n = min(gt_n, pred_n)
            if compare_n == 0:
                molecule_position_rmse.append(0.0)
            else:
                error = (
                    pred_position[idx, :compare_n] - batch["peak_position"][idx, :compare_n]
                ).float()
                molecule_position_rmse.append(float(torch.sqrt((error**2).mean())))
            if gt_n:
                height_correct += (
                    (pred_height[idx, :gt_n] == batch["peak_height"][idx, :gt_n]).sum().item()
                )
                height_total += gt_n
            else:
                height_correct += position.shape[1]
                height_total += position.shape[1]
            if gt_n > 1 and transition is not None:
                pred_transition = transition[idx, : gt_n - 1].argmax(dim=-1)
                gt_transition = (
                    batch["peak_height"][idx, 1:gt_n] != batch["peak_height"][idx, : gt_n - 1]
                ).long()
                transition_correct += (pred_transition == gt_transition).sum().item()
                transition_total += gt_n - 1
            base_id = int(batch["base_id"][idx])
            pair_height_predictions.setdefault(base_id, []).append(
                pred_height[idx, :gt_n].cpu().tolist()
            )
    complete_pairs = [values for values in pair_height_predictions.values() if len(values) == 2]
    complement = []
    for first, second in complete_pairs:
        n = min(len(first), len(second))
        if n:
            complement.extend((int(first[i] != second[i]) for i in range(n)))
    return ECDMetrics(
        loss=total_loss / n_molecules,
        number_rmse=math.sqrt(number_sq_error / n_molecules),
        position_rmse=float(np.mean(molecule_position_rmse)),
        height_accuracy=height_correct / max(height_total, 1),
        transition_accuracy=transition_correct / max(transition_total, 1),
        pair_height_complement_rate=float(np.mean(complement)) if complement else float("nan"),
    )
