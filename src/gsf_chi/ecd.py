from __future__ import annotations

import math

import torch
import torch.nn as nn

import gsf_chi.model as model

N_PEAK_SLOTS = 7
N_POSITION_CLASSES = 20


class PredictionHead(nn.Module):
    def __init__(self, d_model: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OddRMSNorm(nn.Module):
    """Parameter-free normalization that preserves f(-x) = -f(x)."""

    def __init__(self, eps: float = 1e-06):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return x * scale


class ParityOddHeightHead(nn.Module):
    """Predict peak signs with logits that swap exactly under a mirror operation.

    The even stream selects a molecule- and peak-dependent readout direction, while
    the odd stream supplies the signed response.  Since the odd projections have no
    bias, score(z_even, -z_odd) = -score(z_even, z_odd).
    """

    def __init__(self, d_model: int, n_slots: int, rank: int, dropout: float):
        super().__init__()
        self.n_slots = n_slots
        self.rank = rank
        self.even_factor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_slots * rank),
        )
        self.odd_norm = OddRMSNorm()
        self.odd_factor = nn.Linear(d_model, n_slots * rank, bias=False)
        self.odd_direct = nn.Linear(d_model, n_slots, bias=False)
        self.position_embedding = nn.Parameter(torch.empty(N_POSITION_CLASSES, rank))
        self.slot_embedding = nn.Parameter(torch.empty(n_slots, rank))
        self.transition = PredictionHead(d_model, (n_slots - 1) * 2, dropout)
        self.transition_position = nn.Linear(2 * rank, 2, bias=False)
        nn.init.normal_(self.position_embedding, std=rank ** (-0.5))
        nn.init.normal_(self.slot_embedding, std=rank ** (-0.5))

    def forward(
        self, z_even: torch.Tensor, z_odd: torch.Tensor, position_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        odd = self.odd_norm(z_odd)
        even_factor = self.even_factor(z_even).reshape(-1, self.n_slots, self.rank)
        position_probability = torch.softmax(position_logits, dim=-1)
        position_context = torch.einsum(
            "bsp,pr->bsr", position_probability, self.position_embedding
        )
        even_factor = even_factor + position_context + self.slot_embedding[None]
        odd_factor = self.odd_factor(odd).reshape(-1, self.n_slots, self.rank)
        score = (even_factor * odd_factor).sum(dim=-1) / math.sqrt(self.rank)
        score = score + self.odd_direct(odd)
        unary_logits = torch.stack([-score, score], dim=-1)
        transition_logits = self.transition(z_even).reshape(-1, self.n_slots - 1, 2)
        adjacent_position = torch.cat([position_context[:, :-1], position_context[:, 1:]], dim=-1)
        transition_logits = transition_logits + self.transition_position(adjacent_position)
        return (unary_logits, transition_logits)


class ECDModel(nn.Module):
    """Shared encoder with projected Number, Position and Symbol heads."""

    def __init__(
        self,
        mode,
        d_model,
        n_heads,
        n_layers,
        dropout,
        chiral_scale_init,
        readout="parity",
        number_readout="categorical",
        stereo_unit_dropout=0.0,
        n_chiral_heads=None,
        position_decode="independent",
        position_decode_temperature=1.0,
        gsf_scope="global",
        condition_unit_axis=True,
        condition_unit_frequency=True,
        learn_pair_gate=True,
        use_phase_initialization=True,
        share_gsf_axis_frequency=False,
        gsf_support_mode="full",
        gsf_support_fraction=0.0,
        gsf_rotary_mode="residual",
        n_peak_slots=7,
    ):
        super().__init__()
        if readout not in {
            "legacy",
            "parity",
            "parity_split",
            "parity_aux",
            "parity_chain",
            "parity_split_chain",
        }:
            raise ValueError(f"Unknown ECD readout: {readout}")
        if number_readout not in {"categorical", "ordinal"}:
            raise ValueError(f"Unknown Number readout: {number_readout}")
        if position_decode not in {"independent", "posterior_mean", "monotonic"}:
            raise ValueError(f"Unknown Position decoder: {position_decode}")
        if n_peak_slots < 2 or position_decode_temperature <= 0:
            raise ValueError("At least two peak slots and a positive temperature are required")
        self.readout = readout
        self.number_readout = number_readout
        self.position_decode = position_decode
        self.position_decode_temperature = position_decode_temperature
        self.n_peak_slots = n_peak_slots
        self.backbone = model.MolecularModel(
            mode=mode,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            chiral_scale_init=chiral_scale_init,
            stereo_unit_dropout=stereo_unit_dropout,
            n_chiral_heads=n_chiral_heads,
            gsf_scope=gsf_scope,
            condition_unit_axis=condition_unit_axis,
            condition_unit_frequency=condition_unit_frequency,
            learn_pair_gate=learn_pair_gate,
            use_phase_initialization=use_phase_initialization,
            share_gsf_axis_frequency=share_gsf_axis_frequency,
            gsf_support_mode=gsf_support_mode,
            gsf_support_fraction=gsf_support_fraction,
            gsf_rotary_mode=gsf_rotary_mode,
        )
        self.backbone.head = nn.Identity()
        number_dim = n_peak_slots if number_readout == "categorical" else n_peak_slots - 1
        self.number = PredictionHead(d_model, number_dim, dropout)
        self.position = PredictionHead(d_model, n_peak_slots * N_POSITION_CLASSES, dropout)
        if readout != "legacy":
            self.height = ParityOddHeightHead(d_model, n_peak_slots, max(d_model // 2, 8), dropout)
            self.symbol_position = (
                PredictionHead(d_model, n_peak_slots * N_POSITION_CLASSES, dropout)
                if readout in {"parity_split", "parity_split_chain"}
                else None
            )
        else:
            self.height = PredictionHead(d_model, n_peak_slots * 2, dropout)
            self.symbol_position = None

    def forward(self, batch):
        if self.readout == "legacy":
            representation = self.backbone.encode(batch)
            return (
                self.number(representation),
                self.position(representation).reshape(-1, self.n_peak_slots, N_POSITION_CLASSES),
                self.height(representation).reshape(-1, self.n_peak_slots, 2),
                None,
            )
        z_even, z_odd = self.parity_representations(batch)
        number = self.number(z_even)
        position = self.position(z_even).reshape(-1, self.n_peak_slots, N_POSITION_CLASSES)
        context = (
            self.symbol_position(z_even).reshape(-1, self.n_peak_slots, N_POSITION_CLASSES)
            if self.symbol_position is not None
            else position
        )
        height, transition = self.height(z_even, z_odd, context)
        if self.readout in {"parity", "parity_split"}:
            transition = None
        return (number, position, height, transition)

    def parity_representations(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            mirror_batch = dict(batch)
            mirror_batch["chi"] = -batch["chi"]
            if "local_chi" in batch:
                mirror_batch["local_chi"] = -batch["local_chi"]
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state(batch["chi"].device) if batch["chi"].is_cuda else None
            )
            z_chi = self.backbone.encode(batch)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, batch["chi"].device)
            z_mirror = self.backbone.encode(mirror_batch)
            return (0.5 * (z_chi + z_mirror), 0.5 * (z_chi - z_mirror))
        batch_size = batch["chi"].shape[0]
        dual_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim > 0 and (value.shape[0] == batch_size):
                dual_batch[key] = torch.cat([value, value], dim=0)
            else:
                dual_batch[key] = value
        dual_batch["chi"] = torch.cat([batch["chi"], -batch["chi"]], dim=0)
        if "local_chi" in batch:
            dual_batch["local_chi"] = torch.cat([batch["local_chi"], -batch["local_chi"]], dim=0)
        z_chi, z_mirror = self.backbone.encode(dual_batch).chunk(2, dim=0)
        return (0.5 * (z_chi + z_mirror), 0.5 * (z_chi - z_mirror))

    def parity_node_representations(
        self, batch: dict, return_graph: bool = False
    ) -> tuple[torch.Tensor, ...]:
        """Return atom-wise even/odd streams with exactly shared stochastic masks."""
        if self.training:
            mirror_batch = dict(batch)
            mirror_batch["chi"] = -batch["chi"]
            if "local_chi" in batch:
                mirror_batch["local_chi"] = -batch["local_chi"]
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = (
                torch.cuda.get_rng_state(batch["chi"].device) if batch["chi"].is_cuda else None
            )
            nodes_chi, node_mask = self.backbone.encode_nodes(batch)
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, batch["chi"].device)
            nodes_mirror, mirror_mask = self.backbone.encode_nodes(mirror_batch)
            if not torch.equal(node_mask, mirror_mask):
                raise ValueError("mirror encoding changed the atom mask")
            result = (0.5 * (nodes_chi + nodes_mirror), 0.5 * (nodes_chi - nodes_mirror), node_mask)
            if not return_graph:
                return result
            denominator = node_mask.sum(dim=1, keepdim=True).clamp_min(1)
            graph_chi = (nodes_chi * node_mask[..., None]).sum(dim=1) / denominator
            graph_mirror = (nodes_mirror * node_mask[..., None]).sum(dim=1) / denominator
            return result + (0.5 * (graph_chi + graph_mirror), 0.5 * (graph_chi - graph_mirror))
        batch_size = batch["chi"].shape[0]
        dual_batch = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.ndim > 0 and (value.shape[0] == batch_size):
                dual_batch[key] = torch.cat([value, value], dim=0)
            else:
                dual_batch[key] = value
        dual_batch["chi"] = torch.cat([batch["chi"], -batch["chi"]], dim=0)
        if "local_chi" in batch:
            dual_batch["local_chi"] = torch.cat([batch["local_chi"], -batch["local_chi"]], dim=0)
        nodes, masks = self.backbone.encode_nodes(dual_batch)
        nodes_chi, nodes_mirror = nodes.chunk(2, dim=0)
        node_mask, mirror_mask = masks.chunk(2, dim=0)
        if not torch.equal(node_mask, mirror_mask):
            raise ValueError("mirror encoding changed the atom mask")
        result = (0.5 * (nodes_chi + nodes_mirror), 0.5 * (nodes_chi - nodes_mirror), node_mask)
        if not return_graph:
            return result
        denominator = node_mask.sum(dim=1, keepdim=True).clamp_min(1)
        graph_chi = (nodes_chi * node_mask[..., None]).sum(dim=1) / denominator
        graph_mirror = (nodes_mirror * node_mask[..., None]).sum(dim=1) / denominator
        return result + (0.5 * (graph_chi + graph_mirror), 0.5 * (graph_chi - graph_mirror))
