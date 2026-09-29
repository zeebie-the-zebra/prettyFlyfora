"""Load Phase-1 reduced connectome tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


@dataclass
class ReducedNetwork:
    neurons: pd.DataFrame
    edges: pd.DataFrame
    root_ids: np.ndarray
    index_of: dict[int, int]
    sensory_idx: np.ndarray
    hidden_idx: np.ndarray
    motor_idx: np.ndarray
    # Positions within sensory_idx for each modality (and optional side).
    sensory_by_modality: dict[str, np.ndarray] = field(default_factory=dict)


def load_network(
    processed_dir: Path | str = DEFAULT_PROCESSED,
    *,
    network: str = "reduced",
    prefer_parquet: bool = True,
) -> ReducedNetwork:
    """Load ``reduced`` or ``full`` Phase-1 network artifacts."""
    if network not in ("reduced", "full"):
        raise ValueError(f"network must be 'reduced' or 'full', got {network!r}")
    processed_dir = Path(processed_dir)
    prefix = "reduced" if network == "reduced" else "full"
    if prefer_parquet and (processed_dir / f"{prefix}_neurons.parquet").exists():
        neurons = pd.read_parquet(processed_dir / f"{prefix}_neurons.parquet")
        edges = pd.read_parquet(processed_dir / f"{prefix}_edges.parquet")
    else:
        neurons = pd.read_csv(processed_dir / f"{prefix}_neurons.csv")
        edges = pd.read_csv(processed_dir / f"{prefix}_edges.csv")
    return _assemble_network(neurons, edges)


def load_reduced_network(
    processed_dir: Path | str = DEFAULT_PROCESSED,
    prefer_parquet: bool = True,
) -> ReducedNetwork:
    return load_network(processed_dir, network="reduced", prefer_parquet=prefer_parquet)


def _assemble_network(neurons: pd.DataFrame, edges: pd.DataFrame) -> ReducedNetwork:
    neurons = neurons.reset_index(drop=True)
    if "modality" not in neurons.columns:
        neurons["modality"] = np.where(
            neurons["role"] == "SENSORY", "visual", "hidden"
        )

    root_ids = neurons["root_id"].astype(np.int64).to_numpy()
    index_of = {int(r): i for i, r in enumerate(root_ids)}

    role = neurons["role"].to_numpy()
    sensory_idx = np.flatnonzero(role == "SENSORY")
    hidden_idx = np.flatnonzero(role == "HIDDEN")
    motor_idx = np.flatnonzero(role == "MOTOR")

    sensory_by_modality: dict[str, np.ndarray] = {}
    sens = neurons.iloc[sensory_idx].reset_index(drop=True)
    for mod, group in sens.groupby(sens["modality"].fillna("unknown")):
        sensory_by_modality[str(mod)] = group.index.to_numpy()

    return ReducedNetwork(
        neurons=neurons,
        edges=edges,
        root_ids=root_ids,
        index_of=index_of,
        sensory_idx=sensory_idx,
        hidden_idx=hidden_idx,
        motor_idx=motor_idx,
        sensory_by_modality=sensory_by_modality,
    )
