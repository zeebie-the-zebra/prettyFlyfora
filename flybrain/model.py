"""
Leaky rate network over the Phase-1 reduced connectome.

Discrete dynamics (per neural timestep)::

    x <- leak * x + (1 - leak) * (W @ r + I_ext)
    r <- relu(x)              # non-negative; relu(0)=0 keeps zero-input rest exact
    r[sensory] <- I_sensory   # externally clamped

Weight construction
-------------------
Signed synapse counts are placed in W[post, pre], then each postsynaptic row is
L1-normalized by its total absolute incoming weight and multiplied by ``gain``
(default 0.8). That keeps per-neuron drive O(1) when presynaptic rates are O(1),
avoids the "crush multi-hop signal to zero" failure mode of aggressive global
spectral scaling on this E-skewed graph, and still contracts recurrent loops
enough for rest stability when gain < 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .io import ReducedNetwork


@dataclass
class StepResult:
    rates: np.ndarray
    voltages: np.ndarray


class RateNetwork:
    def __init__(
        self,
        net: ReducedNetwork,
        *,
        leak: float = 0.85,
        gain: float = 0.8,
        rate_clip: float | None = 20.0,
    ) -> None:
        if not 0.0 < leak < 1.0:
            raise ValueError("leak must be in (0, 1)")
        if gain <= 0.0:
            raise ValueError("gain must be positive")

        self.net = net
        self.n = len(net.root_ids)
        self.leak = float(leak)
        self.gain = float(gain)
        self.rate_clip = rate_clip

        self.W = self._build_weight_matrix(net, self.gain)
        self.voltage = np.zeros(self.n, dtype=np.float64)
        self.rates = np.zeros(self.n, dtype=np.float64)
        self.I_ext = np.zeros(self.n, dtype=np.float64)
        self._sensory_rates = np.zeros(len(net.sensory_idx), dtype=np.float64)

    @staticmethod
    def _build_weight_matrix(net: ReducedNetwork, gain: float) -> sparse.csr_matrix:
        edges = net.edges
        pre = edges["pre_root_id"].astype(np.int64).to_numpy()
        post = edges["post_root_id"].astype(np.int64).to_numpy()
        w = edges["signed_weight"].astype(np.float64).to_numpy()

        rows = np.fromiter((net.index_of[int(p)] for p in post), dtype=np.int64, count=len(post))
        cols = np.fromiter((net.index_of[int(p)] for p in pre), dtype=np.int64, count=len(pre))
        W = sparse.csr_matrix((w, (rows, cols)), shape=(len(net.root_ids), len(net.root_ids)))
        W.setdiag(0.0)
        W.eliminate_zeros()

        # Row L1-normalize by absolute incoming weight, then apply gain.
        abs_row_sum = np.asarray(np.abs(W).sum(axis=1)).ravel()
        scale = np.zeros_like(abs_row_sum)
        nz = abs_row_sum > 0
        scale[nz] = gain / abs_row_sum[nz]
        W = sparse.diags(scale) @ W

        W._fly_gain = gain  # type: ignore[attr-defined]
        W._fly_n_rows_scaled = int(nz.sum())  # type: ignore[attr-defined]
        W._fly_abs_row_sum_median = float(np.median(abs_row_sum[nz])) if nz.any() else 0.0  # type: ignore[attr-defined]
        return W

    @property
    def weight_gain(self) -> float:
        return float(getattr(self.W, "_fly_gain", self.gain))

    def reset(self) -> None:
        self.voltage[:] = 0.0
        self.rates[:] = 0.0
        self.I_ext[:] = 0.0
        self._sensory_rates[:] = 0.0

    def set_sensory_rates(self, rates: np.ndarray | dict[int, float]) -> None:
        """Clamp SENSORY firing rates for the next steps.

        Accepts either a length-|SENSORY| vector in ``net.sensory_idx`` order,
        or a ``{root_id: rate}`` mapping (unmentioned sensory → 0).
        """
        if isinstance(rates, dict):
            vec = np.zeros(len(self.net.sensory_idx), dtype=np.float64)
            for root_id, val in rates.items():
                idx = self.net.index_of[int(root_id)]
                pos = np.where(self.net.sensory_idx == idx)[0]
                if len(pos) != 1:
                    raise KeyError(f"root_id {root_id} is not a SENSORY neuron")
                vec[pos[0]] = float(val)
            self._sensory_rates = vec
        else:
            rates = np.asarray(rates, dtype=np.float64)
            if rates.shape != (len(self.net.sensory_idx),):
                raise ValueError(
                    f"sensory rates shape {rates.shape} != ({len(self.net.sensory_idx)},)"
                )
            self._sensory_rates = rates.copy()

    def set_sensory_by_side(self, left: float = 0.0, right: float = 0.0) -> None:
        """Legacy helper: set ALL sensory neurons by side (ignores modality)."""
        sides = self.net.neurons.loc[self.net.sensory_idx, "side"].to_numpy()
        vec = np.where(sides == "left", left, 0.0)
        vec = np.where(sides == "right", right, vec)
        self._sensory_rates = vec.astype(np.float64)

    def set_sensory_modalities(
        self,
        drives: dict[str, tuple[float, float] | float],
        *,
        clear: bool = True,
        spatial_gradient: bool = True,
        clamp_max: float = 2.5,
    ) -> None:
        """Set SENSORY rates by modality.

        ``drives`` maps modality name → (left_rate, right_rate) or a scalar
        applied to both sides. Unknown modalities are ignored. When
        ``spatial_gradient`` is True, rates vary across neurons within a
        hemifield so HIDDEN sees patterned input.
        """
        sens = self.net.neurons.iloc[self.net.sensory_idx]
        sides = sens["side"].to_numpy()
        mods = sens["modality"].fillna("").astype(str).to_numpy()
        if clear:
            vec = np.zeros(len(self.net.sensory_idx), dtype=np.float64)
        else:
            vec = self._sensory_rates.copy()

        for mod, val in drives.items():
            if isinstance(val, (tuple, list)):
                left_r, right_r = float(val[0]), float(val[1])
            else:
                left_r = right_r = float(val)
            mask = mods == mod
            if not mask.any():
                continue
            for side, rate in (("left", left_r), ("right", right_r)):
                m = mask & (sides == side)
                idx = np.flatnonzero(m)
                if len(idx) == 0:
                    continue
                if spatial_gradient and len(idx) > 1:
                    g = np.linspace(1.15, 0.75, len(idx))
                    vec[idx] = np.clip(rate * g, 0.0, clamp_max)
                else:
                    vec[idx] = np.clip(rate, 0.0, clamp_max)
        self._sensory_rates = vec

    def step(self) -> StepResult:
        sidx = self.net.sensory_idx
        self.rates[sidx] = self._sensory_rates

        drive = self.W @ self.rates + self.I_ext
        self.voltage = self.leak * self.voltage + (1.0 - self.leak) * drive
        self.rates = np.maximum(0.0, self.voltage)
        if self.rate_clip is not None:
            np.minimum(self.rates, self.rate_clip, out=self.rates)

        self.rates[sidx] = self._sensory_rates
        return StepResult(rates=self.rates, voltages=self.voltage)

    def run(
        self,
        n_steps: int,
        *,
        store: str = "all",
    ) -> dict[str, np.ndarray]:
        """Run ``n_steps`` forward.

        ``store``:
          - ``all``: full (T, N) rate matrix (fine for reduced network)
          - ``roles``: only sensory/hidden/motor trajectories
          - ``motor_only``: only motor trajectory + streaming max stats
        """
        n_steps = int(n_steps)
        store = store if store in ("all", "roles", "motor_only") else "all"

        motor_t = np.zeros((n_steps, len(self.net.motor_idx)), dtype=np.float64)
        max_abs = 0.0
        early_sq = late_sq = 0.0

        if store == "all":
            all_rates = np.zeros((n_steps, self.n), dtype=np.float64)
        elif store == "roles":
            all_rates = None
            sensory_t = np.zeros((n_steps, len(self.net.sensory_idx)), dtype=np.float64)
            hidden_t = np.zeros((n_steps, len(self.net.hidden_idx)), dtype=np.float64)
        else:
            all_rates = sensory_t = hidden_t = None

        for t in range(n_steps):
            self.step()
            r = self.rates
            motor_t[t] = r[self.net.motor_idx]
            max_abs = max(max_abs, float(np.max(np.abs(r))))
            if t < 20:
                early_sq += float(np.mean(r ** 2))
            if t >= n_steps - 20:
                late_sq += float(np.mean(r ** 2))
            if store == "all":
                all_rates[t] = r
            elif store == "roles":
                sensory_t[t] = r[self.net.sensory_idx]
                hidden_t[t] = r[self.net.hidden_idx]

        out: dict[str, np.ndarray | float] = {
            "motor": motor_t,
            "max_abs": max_abs,
            "early_mean_sq": early_sq / min(20, n_steps),
            "late_mean_sq": late_sq / min(20, n_steps),
        }
        if store == "all":
            out["rates"] = all_rates
            out["sensory"] = all_rates[:, self.net.sensory_idx]
            out["hidden"] = all_rates[:, self.net.hidden_idx]
        elif store == "roles":
            out["sensory"] = sensory_t
            out["hidden"] = hidden_t
        return out  # type: ignore[return-value]
