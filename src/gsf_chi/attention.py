from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stereo_unit_dropout_keep(
    rho: torch.Tensor, probability: float, pairwise: bool = False
) -> torch.Tensor:
    """Sample unit-dropout masks, optionally sharing each enantiomer pair."""
    if not 0.0 <= probability < 1.0:
        raise ValueError("stereo-unit dropout probability must be in [0, 1)")
    if pairwise:
        if len(rho) % 2:
            raise ValueError("pairwise stereo-unit dropout requires complete pairs")
        pair_keep = (torch.rand_like(rho[0::2]) >= probability).to(rho.dtype)
        return pair_keep.repeat_interleave(2, dim=0)
    return (torch.rand_like(rho) >= probability).to(rho.dtype)


def _apply_stereo_unit_dropout(
    rho: torch.Tensor,
    chi: torch.Tensor,
    probability: float,
    pairwise: bool = False,
    preserve_chi_magnitude: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply unit dropout with either legacy or physical angle scaling."""
    keep = _stereo_unit_dropout_keep(rho, probability, pairwise=pairwise)
    dropped_rho = rho * keep / (1.0 - probability)
    chi_scale = 1.0 if preserve_chi_magnitude else 1.0 / (1.0 - probability)
    dropped_chi = chi * keep * chi_scale
    return (dropped_rho, dropped_chi)


class GSFChiRoPELayer(nn.Module):
    """Add unit-summed qᵀ(R−I)k corrections to graph attention.

    The field reads the fixed even atom embeddings. Handedness only changes
    the rotation angle; values remain unrotated. Incomplete channel triples
    retain ordinary attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rel_dim: int,
        unit_desc_dim: int,
        n_chiral_heads: int,
        dropout: float,
        chiral_scale_init: float,
        gsf_scope: str = "global",
        condition_unit_axis: bool = True,
        condition_unit_frequency: bool = True,
        learn_pair_gate: bool = True,
        use_phase_initialization: bool = True,
        gsf_support_mode: str = "full",
        gsf_support_fraction: float = 0.0,
        gsf_rotary_mode: str = "residual",
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 1 <= n_chiral_heads <= n_heads:
            raise ValueError("n_chiral_heads must be between 1 and n_heads")
        if gsf_scope not in {"global", "query_anchor", "incident_anchor"}:
            raise ValueError(f"Unknown GSF interaction scope: {gsf_scope}")
        if gsf_support_mode not in {"full", "fixed", "learned_topk"}:
            raise ValueError(f"Unknown GSF support mode: {gsf_support_mode}")
        if gsf_rotary_mode not in {"residual", "direct_replace"}:
            raise ValueError(f"Unknown GSF rotary mode: {gsf_rotary_mode}")
        if not 0.0 <= gsf_support_fraction <= 1.0:
            raise ValueError("GSF support fraction must be in [0, 1]")
        head_dim = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_chiral_heads = n_chiral_heads
        self.gsf_scope = gsf_scope
        self.condition_unit_axis = bool(condition_unit_axis)
        self.condition_unit_frequency = bool(condition_unit_frequency)
        self.learn_pair_gate = bool(learn_pair_gate)
        self.use_phase_initialization = bool(use_phase_initialization)
        self.gsf_support_mode = gsf_support_mode
        self.gsf_support_fraction = float(gsf_support_fraction)
        self.gsf_rotary_mode = gsf_rotary_mode
        self.n_blocks = head_dim // 3
        if self.n_blocks == 0:
            raise ValueError("chiral heads require at least one 3D SO(3) block")
        self.chiral_dim = 3 * self.n_blocks
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.pair_bias = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, n_heads))
        self.unit_encoder = nn.Sequential(
            nn.LayerNorm(unit_desc_dim),
            nn.Linear(unit_desc_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        field_input_dim = 2 * d_model + rel_dim
        self.field = nn.Sequential(
            nn.Linear(field_input_dim, d_model), nn.GELU(), nn.Linear(d_model, 2)
        )
        self.axis_field = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_chiral_heads * self.n_blocks * 3)
        )
        self.frequency_field = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, n_chiral_heads * self.n_blocks)
        )
        self.pair_gate = nn.Sequential(
            nn.Linear(2 * rel_dim + 3, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        nn.init.zeros_(self.axis_field[-1].weight)
        nn.init.zeros_(self.axis_field[-1].bias)
        nn.init.zeros_(self.frequency_field[-1].weight)
        nn.init.zeros_(self.frequency_field[-1].bias)
        axes = torch.randn(n_chiral_heads, self.n_blocks, 3)
        self.axes = nn.Parameter(F.normalize(axes, dim=-1))
        init_omega = torch.logspace(0, -2, self.n_blocks).repeat(n_chiral_heads, 1)
        self.raw_omega = nn.Parameter(torch.log(torch.expm1(init_omega)))
        self.chiral_scale = nn.Parameter(torch.full((n_chiral_heads,), chiral_scale_init))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.capture_diagnostics = False
        self.last_phase: torch.Tensor | None = None
        self.last_pair_gate: torch.Tensor | None = None
        self.last_unit_correction: torch.Tensor | None = None
        self.last_chiral_correction: torch.Tensor | None = None
        self.last_chiral_logit_penalty: torch.Tensor | None = None
        self.last_field_smoothness: torch.Tensor | None = None
        self.intervention_pair_mask: torch.Tensor | None = None

    def _learned_topk_support(
        self,
        score: torch.Tensor,
        anchor: torch.Tensor,
        node_mask: torch.Tensor,
        unit_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select a parity-even hard top-k support with a fixed edge budget.

        With fraction zero, each unit receives exactly the query-anchor entry
        count for that molecule.  The hard membership is selected from the
        learned parity-even gate; gradients still flow through retained gate
        values, while the discrete support itself is treated as a selector.
        """
        valid_pair = (
            node_mask[:, None, :, None] & node_mask[:, None, None, :] & unit_mask[:, :, None, None]
        )
        support = torch.zeros_like(score)
        detached_score = score.detach().masked_fill(~valid_pair, -torch.inf)
        for batch_index in range(score.shape[0]):
            n_atoms = int(node_mask[batch_index].sum().item())
            for unit_index in range(score.shape[1]):
                if not bool(unit_mask[batch_index, unit_index]):
                    continue
                valid_count = n_atoms * n_atoms
                if self.gsf_support_fraction > 0.0:
                    keep_count = int(round(self.gsf_support_fraction * valid_count))
                    keep_count = min(max(keep_count, 1), valid_count)
                else:
                    anchor_count = int(anchor[batch_index, unit_index, :n_atoms].sum().item())
                    keep_count = max(anchor_count * n_atoms, 1)
                flat_score = detached_score[batch_index, unit_index, :n_atoms, :n_atoms].reshape(-1)
                selected = torch.topk(flat_score, k=keep_count, sorted=False).indices
                flat_support = support[batch_index, unit_index, :n_atoms, :n_atoms].reshape(-1)
                flat_support[selected] = 1.0
        return support

    def _chiral_logits(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        even_h: torch.Tensor,
        pair: torch.Tensor,
        rel: torch.Tensor,
        unit_desc: torch.Tensor,
        phase_init: torch.Tensor,
        chi: torch.Tensor,
        rho: torch.Tensor,
        node_mask: torch.Tensor,
        unit_mask: torch.Tensor,
        support_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, n_units, n_atoms, _ = rel.shape
        h_for_units = even_h[:, None].expand(-1, n_units, -1, -1)
        unit_h = self.unit_encoder(unit_desc)
        phase_unit_h = unit_h
        unit_for_atoms = phase_unit_h[:, :, None].expand(-1, -1, n_atoms, -1)
        field_input = torch.cat([h_for_units, unit_for_atoms, rel], dim=-1)
        field_out = self.field(field_input)
        phase_base = phase_init if self.use_phase_initialization else torch.zeros_like(phase_init)
        phase_field = field_out[..., 0]
        psi = phase_base + math.pi * torch.tanh(phase_field)
        node_gate = torch.sigmoid(field_out[..., 1])
        valid_node = node_mask[:, None, :].to(psi.dtype)
        valid_unit = unit_mask[:, :, None].to(psi.dtype)
        denom = valid_node.sum(dim=-1, keepdim=True).clamp_min(1.0)
        psi = (psi - (psi * valid_node).sum(dim=-1, keepdim=True) / denom) * valid_node
        field_difference = psi[:, :, :, None] - psi[:, :, None, :]
        bond_mask = pair[..., 2][:, None] * valid_unit[:, :, :, None]
        bond_mask = bond_mask * valid_node[:, :, :, None] * valid_node[:, :, None, :]
        self.last_field_smoothness = (
            field_difference.square() * bond_mask
        ).sum() / bond_mask.sum().clamp_min(1.0)
        dpsi = psi[:, :, None, :] - psi[:, :, :, None]
        base_omega = F.softplus(self.raw_omega)
        if self.condition_unit_frequency:
            frequency_delta = self.frequency_field(unit_h).reshape(
                bsz, n_units, self.n_chiral_heads, self.n_blocks
            )
        else:
            frequency_delta = unit_h.new_zeros(bsz, n_units, self.n_chiral_heads, self.n_blocks)
        omega = base_omega[None, None] * torch.exp(0.5 * torch.tanh(frequency_delta))
        theta = (
            chi[:, :, None, None, None, None]
            * dpsi[:, :, None, :, :, None]
            * omega[:, :, :, None, None, :]
        )
        q3 = q[:, -self.n_chiral_heads :, :, : self.chiral_dim].reshape(
            bsz, self.n_chiral_heads, n_atoms, self.n_blocks, 3
        )
        k3 = k[:, -self.n_chiral_heads :, :, : self.chiral_dim].reshape(
            bsz, self.n_chiral_heads, n_atoms, self.n_blocks, 3
        )
        if self.condition_unit_axis:
            axis_delta = self.axis_field(unit_h).reshape(
                bsz, n_units, self.n_chiral_heads, self.n_blocks, 3
            )
        else:
            axis_delta = unit_h.new_zeros(bsz, n_units, self.n_chiral_heads, self.n_blocks, 3)
        axis = F.normalize(self.axes[None, None] + 0.5 * torch.tanh(axis_delta), dim=-1)
        dot = torch.einsum("bhimc,bhjmc->bhijm", q3, k3)
        qa = torch.einsum("bhimc,bshmc->bshim", q3, axis)
        ka = torch.einsum("bhjmc,bshmc->bshjm", k3, axis)
        perpendicular = dot[:, None] - torch.einsum("bshim,bshjm->bshijm", qa, ka)
        axis_cross_k = torch.cross(axis[:, :, :, None], k3[:, None], dim=-1)
        oriented = torch.einsum("bhimc,bshjmc->bshijm", q3, axis_cross_k)
        term = (torch.cos(theta) - 1.0) * perpendicular + torch.sin(theta) * oriented
        node_pair_gate = torch.sqrt(
            node_gate[:, :, :, None].clamp_min(1e-08) * node_gate[:, :, None, :].clamp_min(1e-08)
        )
        rel_i = rel[:, :, :, None, :].expand(-1, -1, -1, n_atoms, -1)
        rel_j = rel[:, :, None, :, :].expand(-1, -1, n_atoms, -1, -1)
        pair_for_units = pair[:, None].expand(-1, n_units, -1, -1, -1)
        if self.learn_pair_gate:
            assert self.pair_gate is not None
            learned_pair_gate = torch.sigmoid(
                self.pair_gate(torch.cat([rel_i, rel_j, pair_for_units], dim=-1))
            ).squeeze(-1)
        else:
            learned_pair_gate = node_pair_gate.new_ones(node_pair_gate.shape)
        gate = node_pair_gate * learned_pair_gate
        anchor = rel[..., -2].clamp(0.0, 1.0)
        if self.gsf_scope == "query_anchor":
            gate = gate * anchor[:, :, :, None]
        elif self.gsf_scope == "incident_anchor":
            incident = torch.maximum(anchor[:, :, :, None], anchor[:, :, None, :])
            gate = gate * incident
        if self.gsf_support_mode == "fixed":
            if support_mask is None:
                raise ValueError("Fixed GSF support requires a batch support mask")
            if support_mask.shape != gate.shape:
                raise ValueError(
                    f"gsf_support_mask must match the padded pair gate: {tuple(support_mask.shape)} != {tuple(gate.shape)}"
                )
            gate = gate * support_mask.to(device=gate.device, dtype=gate.dtype)
        elif self.gsf_support_mode == "learned_topk":
            gate = gate * self._learned_topk_support(gate, anchor, node_mask, unit_mask)
        if self.intervention_pair_mask is not None:
            intervention = self.intervention_pair_mask
            if intervention.shape != gate.shape:
                raise ValueError(
                    f"intervention_pair_mask must match the padded pair gate: {tuple(intervention.shape)} != {tuple(gate.shape)}"
                )
            gate = gate * intervention.to(device=gate.device, dtype=gate.dtype)
        weighted = (
            term.sum(dim=-1)
            / math.sqrt(3.0 * self.n_blocks)
            * gate[:, :, None]
            * rho[:, :, None, None, None]
            * valid_unit[:, :, None, None]
        )
        count = unit_mask.sum(dim=1).clamp_min(1).sqrt().to(weighted.dtype)
        correction = weighted.sum(dim=1) / count[:, None, None, None]
        scaled_correction = correction * self.chiral_scale[None, :, None, None]
        output_logits = scaled_correction
        if self.gsf_rotary_mode == "direct_replace":
            rotated_block_score = (dot[:, None] + term).sum(dim=-1)
            direct_weighted = (
                rotated_block_score
                / math.sqrt(self.head_dim)
                * gate[:, :, None]
                * rho[:, :, None, None, None]
                * unit_mask[:, :, None, None, None].to(gate.dtype)
            )
            direct_score = direct_weighted.sum(dim=1) / count[:, None, None, None]
            direct_score = direct_score * self.chiral_scale[None, :, None, None]
            if self.chiral_dim < self.head_dim:
                tail_score = torch.einsum(
                    "bhid,bhjd->bhij",
                    q[:, -self.n_chiral_heads :, :, self.chiral_dim :],
                    k[:, -self.n_chiral_heads :, :, self.chiral_dim :],
                )
            else:
                tail_score = direct_score.new_zeros(direct_score.shape)
            output_logits = direct_score + tail_score / math.sqrt(self.head_dim)
            ordinary_head_score = torch.einsum(
                "bhid,bhjd->bhij", q[:, -self.n_chiral_heads :], k[:, -self.n_chiral_heads :]
            ) / math.sqrt(self.head_dim)
            scaled_correction = output_logits - ordinary_head_score
        if self.capture_diagnostics:
            self.last_phase = psi.detach()
            self.last_pair_gate = gate.detach()
            self.last_unit_correction = (
                weighted
                / count[:, None, None, None, None]
                * self.chiral_scale[None, None, :, None, None]
            ).detach()
            self.last_chiral_correction = scaled_correction.detach()
        self.last_chiral_logit_penalty = scaled_correction.square().mean()
        return output_logits

    def forward(
        self,
        h: torch.Tensor,
        even_h: torch.Tensor,
        pair: torch.Tensor,
        rel: torch.Tensor,
        unit_desc: torch.Tensor,
        phase_init: torch.Tensor,
        chi: torch.Tensor,
        rho: torch.Tensor,
        node_mask: torch.Tensor,
        unit_mask: torch.Tensor,
        use_gsf: bool,
        support_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, n_atoms, _ = h.shape
        self.last_chiral_logit_penalty = h.new_zeros(())
        self.last_field_smoothness = h.new_zeros(())
        qkv = self.qkv(h).reshape(bsz, n_atoms, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        logits = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.head_dim)
        pair_bias = self.pair_bias(pair).permute(0, 3, 1, 2)
        logits = logits + pair_bias
        if use_gsf and unit_mask.any():
            rotary_logits = self._chiral_logits(
                q,
                k,
                even_h,
                pair,
                rel,
                unit_desc,
                phase_init,
                chi,
                rho,
                node_mask,
                unit_mask,
                support_mask,
            )
            if self.gsf_rotary_mode == "direct_replace":
                logits[:, -self.n_chiral_heads :] = (
                    rotary_logits + pair_bias[:, -self.n_chiral_heads :]
                )
            else:
                logits[:, -self.n_chiral_heads :] += rotary_logits
        logits = logits.masked_fill(~node_mask[:, None, None, :], -10000.0)
        attn = self.dropout(torch.softmax(logits, dim=-1))
        update = (
            torch.einsum("bhij,bhjd->bhid", attn, v)
            .transpose(1, 2)
            .reshape(bsz, n_atoms, self.d_model)
        )
        h = self.norm1(h + self.dropout(self.out(update)))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h * node_mask[..., None]
