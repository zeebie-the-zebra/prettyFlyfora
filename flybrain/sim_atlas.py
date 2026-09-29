"""Full-coverage sim atlas: every DN type → decode role + confidence tier.

Curated shortlists stay the high-confidence slice. This module maps *all*
types (curated + putative + parked-with-stim + prefix fallback) into
sim-facing roles so the motor decoder can drive locomotion from the full
MOTOR population — not only the partner-validated shortlist.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .motor_decode import _DN_ROLE, _role_for_type

# Decode-facing roles understood by MotorDecoder (ethology + legacy).
SIM_ROLES = frozenset(
    {
        "escape",
        "wind",
        "odor",
        "taste",
        "thrust",
        "steer",
        "brake",
        "arousal",
        "walk",
        "generic",
        "silent",
    }
)

# How strongly a type's activity scales static decode weight.
TIER_SCALE = {
    "hand": 1.0,
    "curated": 1.0,
    "putative": 0.60,
    "stim": 0.45,
    "prefix": 0.35,
    "silent": 0.0,
}

# Partner-validated / putative curated_role → (sim_role, tier, scale)
_CURATED_TO_SIM: dict[str, tuple[str, str, float]] = {
    "escape": ("escape", "curated", 1.0),
    "wind_jo": ("wind", "curated", 1.0),
    "taste_contact": ("taste", "curated", 1.0),
    "odor_al": ("odor", "curated", 1.0),
    "wind_putative": ("wind", "putative", 0.60),
    "odor_putative": ("odor", "putative", 0.60),
    "silent_pool": ("silent", "silent", 0.0),
}

_STIM_SCALE = {"strong": 0.50, "mod": 0.40, "weak": 0.30}


def assign_sim_role(row: pd.Series) -> dict:
    """Map one detailed-atlas type row to sim_role / tier / weight_scale."""
    ptype = str(row.get("primary_type", "") or "")
    curated = str(row.get("curated_role", "") or "")
    primary = str(row.get("primary_modality", "none") or "none")
    p_tier = str(row.get("primary_tier", "none") or "none")
    lr = str(row.get("lr_class", "none") or "none")
    tags = str(row.get("tags", "") or "")
    decoder = str(row.get("decoder_role", "") or _role_for_type(ptype))

    # 1) Hand-mapped exact types (legacy reduced DN catalogue).
    if ptype in _DN_ROLE:
        role = _DN_ROLE[ptype]
        return {
            "sim_role": role,
            "sim_tier": "hand",
            "sim_weight_scale": TIER_SCALE["hand"],
            "sim_source": f"hand _DN_ROLE={role}",
        }

    # 2) Curated / putative ethology shortlists (includes silent_pool).
    if curated in _CURATED_TO_SIM:
        role, tier, scale = _CURATED_TO_SIM[curated]
        return {
            "sim_role": role,
            "sim_tier": tier,
            "sim_weight_scale": scale,
            "sim_source": f"curated_role={curated}",
        }

    # 3) Parked but modality-active → ethology or locomotion from fingerprint.
    if primary in ("loom", "wind", "odor", "taste") and p_tier in _STIM_SCALE:
        role = {
            "loom": "escape",
            "wind": "wind",
            "odor": "odor",
            "taste": "taste",
        }[primary]
        return {
            "sim_role": role,
            "sim_tier": "stim",
            "sim_weight_scale": _STIM_SCALE[p_tier],
            "sim_source": f"parked primary={primary}.{p_tier}",
        }

    if primary == "visual" and p_tier in _STIM_SCALE:
        if lr == "lateralized" or "turn" in tags.split("+"):
            return {
                "sim_role": "steer",
                "sim_tier": "stim",
                "sim_weight_scale": _STIM_SCALE[p_tier],
                "sim_source": f"visual.{p_tier} lateralized/turn",
            }
        if lr == "bilateral" or "walk" in tags.split("+"):
            return {
                "sim_role": "walk",
                "sim_tier": "stim",
                "sim_weight_scale": _STIM_SCALE[p_tier],
                "sim_source": f"visual.{p_tier} bilateral/walk",
            }
        return {
            "sim_role": "thrust" if decoder == "thrust" else "steer",
            "sim_tier": "stim",
            "sim_weight_scale": _STIM_SCALE[p_tier] * 0.9,
            "sim_source": f"visual.{p_tier} → {decoder or 'steer'}",
        }

    # 4) Prefix / decoder_role fallback (covers remaining parked).
    if decoder in ("thrust", "steer", "brake", "arousal"):
        return {
            "sim_role": decoder,
            "sim_tier": "prefix",
            "sim_weight_scale": TIER_SCALE["prefix"],
            "sim_source": f"decoder_role={decoder}",
        }

    return {
        "sim_role": "generic",
        "sim_tier": "prefix",
        "sim_weight_scale": 0.25,
        "sim_source": "fallback generic",
    }


def build_sim_atlas(detailed: pd.DataFrame) -> pd.DataFrame:
    """Assign sim_role to every type in the detailed atlas (full coverage)."""
    if detailed is None or len(detailed) == 0:
        return pd.DataFrame(
            columns=[
                "primary_type",
                "sim_role",
                "sim_tier",
                "sim_weight_scale",
                "sim_source",
            ]
        )

    rows = []
    for _, row in detailed.iterrows():
        assigned = assign_sim_role(row)
        rows.append(
            {
                "primary_type": str(row.get("primary_type", "")),
                **assigned,
                "curated_role": str(row.get("curated_role", "")),
                "map_code": str(row.get("map_code", "")),
                "tags": str(row.get("tags", "")),
                "primary_modality": str(row.get("primary_modality", "")),
                "primary_tier": str(row.get("primary_tier", "")),
                "decoder_role": str(row.get("decoder_role", "")),
                "discovery_role": str(row.get("discovery_role", "")),
                "discovery_confidence": float(row.get("discovery_confidence", 0.0) or 0.0),
                "n_instances": int(row.get("n_instances", 0) or 0),
            }
        )
    out = pd.DataFrame(rows)
    # Stable column order
    front = [
        "primary_type",
        "sim_role",
        "sim_tier",
        "sim_weight_scale",
        "sim_source",
        "curated_role",
        "map_code",
        "tags",
        "primary_modality",
        "primary_tier",
        "decoder_role",
        "discovery_role",
        "discovery_confidence",
        "n_instances",
    ]
    return out[front]


def write_sim_atlas(sim: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    """Write dn_sim_atlas.csv + role/tier histograms."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = out_dir / "dn_sim_atlas.csv"
    sim.to_csv(atlas_path, index=False)

    role_hist = (
        sim["sim_role"].value_counts().rename_axis("sim_role").reset_index(name="n_types")
    )
    tier_hist = (
        sim["sim_tier"].value_counts().rename_axis("sim_tier").reset_index(name="n_types")
    )
    role_path = out_dir / "dn_sim_role_histogram.csv"
    tier_path = out_dir / "dn_sim_tier_histogram.csv"
    role_hist.to_csv(role_path, index=False)
    tier_hist.to_csv(tier_path, index=False)
    return {
        "sim_atlas": atlas_path,
        "sim_role_histogram": role_path,
        "sim_tier_histogram": tier_path,
    }


def load_sim_role_table(
    processed_dir: Path | str,
    network: str = "full",
) -> dict[str, tuple[str, float]] | None:
    """Load primary_type → (sim_role, weight_scale) from dn_sim_atlas.csv.

    Returns None if the atlas file is missing.
    """
    path = Path(processed_dir) / "phase2" / network / "dn_sim_atlas.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "primary_type" not in df.columns or "sim_role" not in df.columns:
        return None
    scale_col = "sim_weight_scale" if "sim_weight_scale" in df.columns else None
    out: dict[str, tuple[str, float]] = {}
    for _, row in df.iterrows():
        ptype = str(row["primary_type"])
        role = str(row["sim_role"])
        if role not in SIM_ROLES:
            role = "generic"
        scale = float(row[scale_col]) if scale_col else TIER_SCALE.get(
            str(row.get("sim_tier", "prefix")), 0.35
        )
        out[ptype] = (role, scale)
    return out
