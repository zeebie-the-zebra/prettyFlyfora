"""
Population-level MOTOR decoder for Phase 3.

Problem
-------
The reduced network had ~12 named descending neurons decoded by hand (DNb05_left →
thrust, DNa03 → steer, …). The full brain has 1,301 MOTOR neurons with duplicate
type_side labels, so per-name logic does not scale.

Proposed aggregation (implemented here)
---------------------------------------
1. **Calibration weights** (static, from Phase-2 ``motor_stimulation_summary.csv``):
   - Join by ``root_id`` (not label — labels collide in full brain).
   - Category filter: dud → 0, weak → 0.35, responsive → 1.0.
   - Optional stim gain: ``g = clip(any_response / median_responsive, 0.1, 2.0)``.
   - Final static weight: ``w_i = category_weight × g``.

2. **Turn** — bilateral weighted vote (all non-dud neurons):
   ``T = Σ w_i · a_i · s_i / (Σ w_i · a_i + ε)`` where ``s_i = +1`` left, ``-1`` right,
   ``a_i`` is relative activity (see below). This is side-agnostic and scales to any
   MOTOR count.

3. **Forward** — role-weighted pool:
   Prefer ``dn_sim_atlas.csv`` (full-coverage sim roles) when present; else legacy
   ``primary_type`` → thrust / steer / brake / arousal / generic.
   Ethology roles (escape / wind / odor / taste / walk) fold into forward with
   distinct gains; steer still drives turn only via the bilateral vote.
   ``F = idle + 0.75·thrust + 0.90·escape + 0.55·wind + 0.60·odor
         + 0.50·taste + 0.55·walk + 0.35·arousal - 0.8·brake + 0.25·generic``.

4. **Relative activity** (when ``relative=True``):
   ``a_i = r_i / max_j(r_j)`` over weighted neurons with ``r_j > 0``; if all zero,
   return idle forward and zero turn.

This preserves the reduced-network intuition (L/R differential steering + role-based
forward/brake) while scaling via population statistics and Phase-2 quality weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .io import ReducedNetwork, DEFAULT_PROCESSED

# Exact-type map plus prefix heuristics for the larger DN catalogue.
_DN_ROLE: dict[str, str] = {
    "DNb05": "thrust",
    "DNg75": "thrust",
    "DNg100": "thrust",
    "DNg74_a": "thrust",
    "DNa03": "steer",
    "DNa02": "steer",
    "DNp103": "steer",
    "DNpe042": "brake",
    "DNpe053": "brake",
    "DNge037": "steer",
    "aSP22": "arousal",
}

# Prefix → role when exact type is unknown (FlyWire primary_type families).
_DN_PREFIX_ROLE: tuple[tuple[str, str], ...] = (
    ("DNa", "steer"),
    ("DNb", "thrust"),
    ("DNg", "thrust"),
    ("DNp", "steer"),
    ("DNpe", "brake"),
    ("DNge", "steer"),
    ("DNc", "thrust"),
    ("aSP", "arousal"),
)


def _role_for_type(primary_type: str) -> str:
    t = str(primary_type)
    if t in _DN_ROLE:
        return _DN_ROLE[t]
    # Longer prefixes first.
    for prefix, role in sorted(_DN_PREFIX_ROLE, key=lambda x: -len(x[0])):
        if t.startswith(prefix):
            return role
    return "generic"


CATEGORY_WEIGHT = {"responsive": 1.0, "weak": 0.35, "dud": 0.0}
DEFAULT_SUMMARY = DEFAULT_PROCESSED / "phase2"


@dataclass
class MotorCalibration:
    """Per-MOTOR-neuron static decode metadata aligned with ``net.motor_idx`` order."""

    root_ids: np.ndarray
    sides: np.ndarray  # "left" | "right" | other
    primary_types: np.ndarray
    roles: np.ndarray  # thrust | steer | brake | arousal | escape | wind | …
    static_weights: np.ndarray  # category × stim gain × sim_weight_scale
    categories: np.ndarray
    n_responsive: int = 0
    n_weak: int = 0
    n_dud: int = 0
    summary_path: Path | None = None
    sim_atlas_path: Path | None = None
    n_sim_mapped: int = 0

    @property
    def n_motor(self) -> int:
        return len(self.root_ids)


@dataclass
class DecodeResult:
    turn: float
    forward: float
    active_weight: float
    n_active: int
    turn_left_mass: float = 0.0
    turn_right_mass: float = 0.0
    thrust: float = 0.0
    brake: float = 0.0
    arousal: float = 0.0
    escape: float = 0.0
    wind: float = 0.0
    odor: float = 0.0
    taste: float = 0.0


def load_motor_calibration(
    net: ReducedNetwork,
    *,
    network: str = "reduced",
    processed_dir: Path | str = DEFAULT_PROCESSED,
    use_sim_atlas: bool = True,
) -> MotorCalibration:
    """Load Phase-2 category weights keyed by root_id; fall back to all-ones.

    When ``dn_sim_atlas.csv`` exists for ``network`` and ``use_sim_atlas`` is True,
    roles + weight scales come from the full-coverage sim atlas (all DN types).
    """
    processed_dir = Path(processed_dir)
    meta = net.neurons.iloc[net.motor_idx].reset_index(drop=True)
    root_ids = meta["root_id"].astype(np.int64).to_numpy()
    sides = meta["side"].astype(str).to_numpy()
    primary_types = meta["primary_type"].astype(str).to_numpy()
    roles = np.array([_role_for_type(t) for t in primary_types], dtype=object)
    sim_scale = np.ones(len(root_ids), dtype=np.float64)
    sim_atlas_path: Path | None = None
    n_sim_mapped = 0

    summary_path = processed_dir / "phase2" / network / "motor_stimulation_summary.csv"
    categories = np.full(len(root_ids), "unknown", dtype=object)
    stim_gain = np.ones(len(root_ids), dtype=np.float64)
    lr_sel = np.zeros(len(root_ids), dtype=np.float64)
    any_resp = np.zeros(len(root_ids), dtype=np.float64)

    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if "root_id" in df.columns:
            cal = df.set_index("root_id")
            for i, rid in enumerate(root_ids):
                if int(rid) not in cal.index:
                    continue
                row = cal.loc[int(rid)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                categories[i] = str(row.get("category", "unknown"))
                ar = float(row.get("any_response", 0.0))
                stim_gain[i] = ar
                any_resp[i] = ar
                if "L_minus_R" in row.index:
                    lr_sel[i] = float(row.get("L_minus_R", 0.0))
        else:
            # Legacy CSV without root_id: match first occurrence per label (reduced only).
            by_label = df.drop_duplicates(subset=["motor"], keep="first").set_index("motor")
            labels = [f"{t}_{s}" for t, s in zip(primary_types, sides)]
            for i, lab in enumerate(labels):
                if lab not in by_label.index:
                    continue
                row = by_label.loc[lab]
                categories[i] = str(row.get("category", "unknown"))
                ar = float(row.get("any_response", 0.0))
                stim_gain[i] = ar
                any_resp[i] = ar
                if "L_minus_R" in row.index:
                    lr_sel[i] = float(row.get("L_minus_R", 0.0))

    # Full-coverage sim atlas overrides prefix roles when present.
    if use_sim_atlas:
        from .sim_atlas import load_sim_role_table

        table = load_sim_role_table(processed_dir, network=network)
        if table is not None:
            sim_atlas_path = processed_dir / "phase2" / network / "dn_sim_atlas.csv"
            for i, ptype in enumerate(primary_types):
                # Keep exact hand map unless atlas explicitly remaps the same type.
                if str(ptype) in _DN_ROLE and str(ptype) not in table:
                    continue
                if str(ptype) not in table:
                    continue
                role, scale = table[str(ptype)]
                if role == "silent":
                    roles[i] = "silent"
                    sim_scale[i] = 0.0
                else:
                    roles[i] = role
                    sim_scale[i] = float(scale)
                n_sim_mapped += 1

    # Test B L/R selectivity: only re-label non-atlas / non-hand types so
    # mapped ethology + hand thrust/brake/arousal DNs stay put.
    for i in range(len(roles)):
        if categories[i] == "dud":
            continue
        ar = any_resp[i]
        if ar < 1e-4:
            continue
        if str(primary_types[i]) in _DN_ROLE:
            continue
        # Skip if sim atlas already assigned a specialized role.
        if roles[i] in (
            "escape",
            "wind",
            "odor",
            "taste",
            "walk",
            "silent",
            "brake",
            "arousal",
        ):
            continue
        if abs(lr_sel[i]) >= 0.25 * ar:
            roles[i] = "steer"
        elif roles[i] == "generic" and categories[i] in ("responsive", "weak"):
            roles[i] = "thrust"

    # Stim gain normalization among responsive neurons.
    resp_mask = categories == "responsive"
    if resp_mask.any():
        med = float(np.median(stim_gain[resp_mask]))
        if med > 0:
            stim_gain = np.clip(stim_gain / med, 0.1, 2.0)

    static = np.array(
        [CATEGORY_WEIGHT.get(str(c), 0.5) for c in categories], dtype=np.float64
    )
    static *= stim_gain
    static *= sim_scale
    # Silent roles contribute neither turn nor forward.
    static = np.where(roles == "silent", 0.0, static)

    n_resp = int((categories == "responsive").sum())
    n_weak = int((categories == "weak").sum())
    n_dud = int((categories == "dud").sum())

    return MotorCalibration(
        root_ids=root_ids,
        sides=sides,
        primary_types=primary_types,
        roles=roles,
        static_weights=static,
        categories=categories,
        n_responsive=n_resp,
        n_weak=n_weak,
        n_dud=n_dud,
        summary_path=summary_path if summary_path.exists() else None,
        sim_atlas_path=sim_atlas_path,
        n_sim_mapped=n_sim_mapped,
    )


class MotorDecoder:
    def __init__(
        self,
        cal: MotorCalibration,
        *,
        relative: bool = True,
        idle_forward: float = 0.28,
    ) -> None:
        self.cal = cal
        self.relative = relative
        self.idle_forward = idle_forward
        self._side_sign = np.where(cal.sides == "left", 1.0, -1.0).astype(np.float64)

    def decode(self, motor_rates: np.ndarray) -> DecodeResult:
        cal = self.cal
        rates = np.asarray(motor_rates, dtype=np.float64).ravel()
        if rates.size != cal.n_motor:
            raise ValueError(f"expected {cal.n_motor} motor rates, got {rates.size}")

        w = cal.static_weights
        mask = w > 0
        if not mask.any():
            return DecodeResult(
                turn=0.0,
                forward=self.idle_forward,
                active_weight=0.0,
                n_active=0,
            )

        wr = w * rates
        active = mask & (rates > 1e-12)
        if not active.any():
            return DecodeResult(
                turn=0.0,
                forward=self.idle_forward,
                active_weight=0.0,
                n_active=0,
            )

        if self.relative:
            peak = float(rates[active].max())
            act = np.where(active, rates / peak, 0.0)
        else:
            act = np.clip(rates, 0.0, 1.0)

        wa = w * act
        turn_den = float(wa[active].sum()) + 1e-12
        turn_num = float((wa * self._side_sign)[active].sum())
        turn = turn_num / turn_den

        left_mass = float(wa[(active) & (cal.sides == "left")].sum())
        right_mass = float(wa[(active) & (cal.sides == "right")].sum())

        thrust = brake = arousal = generic = 0.0
        escape = wind = odor = taste = walk = 0.0
        thrust_w = brake_w = arousal_w = generic_w = 0.0
        escape_w = wind_w = odor_w = taste_w = walk_w = 0.0
        for i in np.flatnonzero(active):
            a = float(act[i])
            wi = float(w[i])
            role = cal.roles[i]
            if role == "thrust":
                thrust += wi * a
                thrust_w += wi
            elif role == "escape":
                escape += wi * a
                escape_w += wi
            elif role == "wind":
                wind += wi * a
                wind_w += wi
            elif role == "odor":
                odor += wi * a
                odor_w += wi
            elif role == "taste":
                taste += wi * a
                taste_w += wi
            elif role == "walk":
                walk += wi * a
                walk_w += wi
            elif role == "brake":
                brake += wi * a
                brake_w += wi
            elif role == "arousal":
                arousal += wi * a
                arousal_w += wi
            elif role == "steer":
                pass  # turn only via bilateral vote
            elif role == "silent":
                pass
            else:
                generic += wi * a
                generic_w += wi

        def _norm(num: float, den: float) -> float:
            return num / den if den > 0 else 0.0

        thrust_m = _norm(thrust, thrust_w)
        escape_m = _norm(escape, escape_w)
        wind_m = _norm(wind, wind_w)
        odor_m = _norm(odor, odor_w)
        taste_m = _norm(taste, taste_w)
        walk_m = _norm(walk, walk_w)
        brake_m = _norm(brake, brake_w)
        arousal_m = _norm(arousal, arousal_w)
        generic_m = _norm(generic, generic_w)

        forward = max(
            0.0,
            self.idle_forward
            + 0.75 * thrust_m
            + 0.90 * escape_m
            + 0.55 * wind_m
            + 0.60 * odor_m
            + 0.50 * taste_m
            + 0.55 * walk_m
            + 0.35 * arousal_m
            - 0.8 * brake_m
            + 0.25 * generic_m,
        )
        forward = float(min(forward, 1.5))

        return DecodeResult(
            turn=float(turn),
            forward=forward,
            active_weight=turn_den,
            n_active=int(active.sum()),
            turn_left_mass=left_mass,
            turn_right_mass=right_mass,
            thrust=thrust_m,
            brake=brake_m,
            arousal=arousal_m,
            escape=escape_m,
            wind=wind_m,
            odor=odor_m,
            taste=taste_m,
        )
