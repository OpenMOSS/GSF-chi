from __future__ import annotations

import torch
import torch.nn as nn

from gsf_chi.attention import GSFChiRoPELayer, _apply_stereo_unit_dropout
from gsf_chi.features import REL_DIM, UNIT_DESC_DIM


class MolecularModel(nn.Module):
    """Mean-pooled Graph Transformer with global, local or token chirality input."""

    def __init__(
        self,
        mode: str,
        input_dim: int = 52,
        d_model: int = 48,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        chiral_scale_init: float = 5.0,
        stereo_unit_dropout: float = 0.0,
        n_chiral_heads: int | None = None,
        gsf_scope: str = "global",
        condition_unit_axis: bool = True,
        condition_unit_frequency: bool = True,
        learn_pair_gate: bool = True,
        use_phase_initialization: bool = True,
        share_gsf_axis_frequency: bool = False,
        gsf_support_mode: str = "full",
        gsf_support_fraction: float = 0.0,
        gsf_rotary_mode: str = "residual",
    ):
        super().__init__()
        if mode not in {"base", "token", "local", "gsf", "gsf_local", "gsf_token"}:
            raise ValueError(mode)
        self.mode = mode
        if not 0.0 <= stereo_unit_dropout < 1.0:
            raise ValueError("stereo_unit_dropout must be in [0, 1)")
        if n_chiral_heads is None:
            n_chiral_heads = n_heads // 2
        if not 1 <= n_chiral_heads <= n_heads:
            raise ValueError("n_chiral_heads must be between 1 and n_heads")
        self.stereo_unit_dropout = stereo_unit_dropout
        self.share_gsf_axis_frequency = bool(share_gsf_axis_frequency)
        self.embed = nn.Sequential(nn.Linear(input_dim, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.chi_embed = nn.Linear(1, d_model, bias=False)
        self.layers = nn.ModuleList(
            [
                GSFChiRoPELayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    rel_dim=REL_DIM,
                    unit_desc_dim=UNIT_DESC_DIM,
                    n_chiral_heads=n_chiral_heads,
                    dropout=dropout,
                    chiral_scale_init=chiral_scale_init,
                    gsf_scope=gsf_scope,
                    condition_unit_axis=condition_unit_axis,
                    condition_unit_frequency=condition_unit_frequency,
                    learn_pair_gate=learn_pair_gate,
                    use_phase_initialization=use_phase_initialization,
                    gsf_support_mode=gsf_support_mode,
                    gsf_support_fraction=gsf_support_fraction,
                    gsf_rotary_mode=gsf_rotary_mode,
                )
                for _ in range(n_layers)
            ]
        )
        reference_layer = self.layers[0]
        if self.share_gsf_axis_frequency:
            for layer in self.layers[1:]:
                layer.axis_field = reference_layer.axis_field
                layer.frequency_field = reference_layer.frequency_field
                layer.axes = reference_layer.axes
                layer.raw_omega = reference_layer.raw_omega
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def encode_nodes(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the final atom tokens and their validity mask.

        Keeping this operation separate from graph pooling lets downstream
        tasks attach structured decoders without changing any GSF-ChiRoPE
        attention layer or its stereochemical field.
        """
        node_mask = batch["node_mask"]
        rho = batch["rho"]
        local_chi = batch["local_chi"]
        effective_chi = batch["chi"]
        if self.training and self.stereo_unit_dropout > 0.0:
            rho, effective_chi = _apply_stereo_unit_dropout(
                rho,
                batch["chi"],
                self.stereo_unit_dropout,
                pairwise=bool(batch.get("_pairwise_stereo_unit_dropout", False)),
                preserve_chi_magnitude=bool(batch.get("_preserve_dropout_chi_magnitude", False)),
            )
            if self.mode in {"local", "gsf_local"}:
                anchors = batch["unit_rel"][..., -2]
                local_chi = (effective_chi[:, :, None] * anchors).sum(dim=1, keepdim=False)[
                    ..., None
                ]
        h = self.embed(batch["node"])
        even_h = h
        if self.mode in {"local", "gsf_local"}:
            h = h + self.chi_embed(local_chi)
        elif self.mode in {"token", "gsf_token"}:
            count = batch["unit_mask"].sum(dim=1, keepdim=True).clamp_min(1).sqrt()
            summary = (effective_chi * batch["unit_mask"]).sum(dim=1, keepdim=True) / count
            h = h + self.chi_embed(summary[:, None].expand(-1, h.shape[1], -1))
        h = h * node_mask[..., None]
        pair = batch["pair"]
        unit_rel = batch["unit_rel"]
        phase_init = batch["phase_init"]
        layer_node_mask = node_mask
        for layer in self.layers:
            h = layer(
                h,
                even_h,
                pair,
                unit_rel,
                batch["unit_desc"],
                phase_init,
                batch["chi"],
                rho,
                layer_node_mask,
                batch["unit_mask"],
                use_gsf=self.mode in {"gsf", "gsf_local", "gsf_token"},
                support_mask=batch.get("gsf_support_mask"),
            )
        return (h, layer_node_mask)

    def encode(self, batch: dict) -> torch.Tensor:
        h, node_mask = self.encode_nodes(batch)
        denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return (h * node_mask[..., None]).sum(dim=1) / denom

    def regularization_terms(self) -> dict[str, torch.Tensor]:
        reference = next(self.parameters())
        chiral_logit = reference.new_zeros(())
        field_smoothness = reference.new_zeros(())
        for layer in self.layers:
            if layer.last_chiral_logit_penalty is not None:
                chiral_logit = chiral_logit + layer.last_chiral_logit_penalty
            if layer.last_field_smoothness is not None:
                field_smoothness = field_smoothness + layer.last_field_smoothness
        return {"chiral_logit": chiral_logit, "field_smoothness": field_smoothness}

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.encode(batch))


class RankingModel(MolecularModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        d_model = self.head[-1].in_features
        self.head[-1] = nn.Linear(d_model, 1)

    def forward(self, batch: dict) -> torch.Tensor:
        return self.head(self.encode(batch)).squeeze(-1)
