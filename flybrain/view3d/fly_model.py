"""Procedural fly mesh + GLTF swap stub."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .anim import AnimState
from .pose import FlyPose


def _transform_points(pts: np.ndarray, mat: np.ndarray) -> np.ndarray:
    hom = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    out = (mat @ hom.T).T[:, :3]
    return out.astype(np.float32)


def _mat_trs(tx: float, ty: float, tz: float, yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """Build TRS. Yaw about Y maps local +X → (cos(yaw), 0, sin(yaw)) to match sim heading."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    # Y-up: local +X faces travel direction (cos θ, 0, sin θ) when yaw=θ.
    ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    r = ry @ rx
    if abs(pitch) > 1e-6:
        r = r @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = r
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def _box(sx: float, sy: float, sz: float, color: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axis-aligned box centered at origin → interleaved positions + flat colors, indices."""
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    corners = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    faces = [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (2, 3, 7, 6),
        (0, 3, 7, 4),
        (1, 2, 6, 5),
    ]
    positions = []
    colors = []
    indices = []
    for face in faces:
        base = len(positions)
        for i in face:
            positions.append(corners[i])
            colors.append(color)
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 3])
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def _ellipsoid(rx: float, ry: float, rz: float, color: tuple[float, float, float], segments: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    colors = []
    indices = []
    for i in range(segments + 1):
        lat = np.pi * (-0.5 + i / segments)
        for j in range(segments + 1):
            lon = 2 * np.pi * (j / segments)
            x = rx * np.cos(lat) * np.cos(lon)
            y = ry * np.sin(lat)
            z = rz * np.cos(lat) * np.sin(lon)
            positions.append([x, y, z])
            colors.append(color)
    def vid(i: int, j: int) -> int:
        return i * (segments + 1) + j

    for i in range(segments):
        for j in range(segments):
            a = vid(i, j)
            b = vid(i, j + 1)
            c = vid(i + 1, j + 1)
            d = vid(i + 1, j)
            indices.extend([a, b, c, a, c, d])
    return (
        np.asarray(positions, dtype=np.float32),
        np.asarray(colors, dtype=np.float32),
        np.asarray(indices, dtype=np.uint32),
    )


def _append_mesh(
    all_pos: list[np.ndarray],
    all_col: list[np.ndarray],
    all_idx: list[np.ndarray],
    pos: np.ndarray,
    col: np.ndarray,
    idx: np.ndarray,
    mat: np.ndarray,
) -> None:
    base = sum(len(p) for p in all_pos)
    all_pos.append(_transform_points(pos, mat))
    all_col.append(col)
    all_idx.append(idx + base)


class FlyModel(ABC):
    @abstractmethod
    def mesh(self, pose: FlyPose, anim: AnimState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (positions Nx3, colors Nx3, indices M) in world space."""


class ProceduralFly(FlyModel):
    """Stylized procedural fruit fly: body, head, wings, six legs."""

    BODY = (0.12, 0.10, 0.08)
    THORAX = (0.18, 0.14, 0.10)
    HEAD = (0.22, 0.12, 0.08)
    EYE = (0.75, 0.15, 0.12)
    WING = (0.55, 0.65, 0.55)
    LEG = (0.15, 0.12, 0.08)

    def mesh(self, pose: FlyPose, anim: AnimState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        root = _mat_trs(pose.x, pose.y, pose.z, yaw=pose.yaw, pitch=pose.pitch, roll=pose.roll)
        # Lying blend: flatten toward ground in local space by scaling Y later via root pitch already.

        all_pos: list[np.ndarray] = []
        all_col: list[np.ndarray] = []
        all_idx: list[np.ndarray] = []

        # Abdomen
        p, c, i = _ellipsoid(0.55, 0.22, 0.22, self.BODY, 8)
        _append_mesh(all_pos, all_col, all_idx, p, c, i, root @ _mat_trs(-0.35, 0.15, 0.0))

        # Thorax
        p, c, i = _ellipsoid(0.32, 0.26, 0.26, self.THORAX, 8)
        _append_mesh(all_pos, all_col, all_idx, p, c, i, root @ _mat_trs(0.15, 0.22, 0.0))

        # Head
        p, c, i = _ellipsoid(0.22, 0.2, 0.22, self.HEAD, 8)
        _append_mesh(all_pos, all_col, all_idx, p, c, i, root @ _mat_trs(0.48, 0.22, 0.0))

        # Eyes
        for side in (-1.0, 1.0):
            p, c, i = _ellipsoid(0.1, 0.12, 0.12, self.EYE, 6)
            _append_mesh(all_pos, all_col, all_idx, p, c, i, root @ _mat_trs(0.55, 0.25, side * 0.16))

        # Wings: local box long in +X; yaw ±90° so they extend sideways from thorax.
        # Flap = roll about body X (lifts wing tips).
        wing_l = anim.wing_angle
        wing_r = -anim.wing_angle
        for side, wang in ((-1.0, wing_l), (1.0, wing_r)):
            p, c, i = _box(0.85, 0.02, 0.28, self.WING)
            flap = 0.2 + wang  # slight dihedral + flap
            hinge = root @ _mat_trs(
                0.05,
                0.36,
                side * 0.1,
                yaw=-side * (0.5 * np.pi),
                roll=side * flap,
            )
            _append_mesh(all_pos, all_col, all_idx, p, c, i, hinge @ _mat_trs(0.42, 0.0, 0.0))

        # Six legs
        leg_zs = [-0.18, 0.0, 0.18]
        for li, z0 in enumerate(leg_zs):
            for side in (-1.0, 1.0):
                phase = anim.leg_phase + li * 2.1 + (0.0 if side < 0 else _LEG_PHASE_OFFSET)
                if anim.limp > 0 and side > 0:
                    phase *= 1.0 - 0.5 * anim.limp
                swing = anim.leg_amp * np.sin(phase)
                lift = abs(anim.leg_amp * np.cos(phase)) * 0.15
                if anim.groom_t > 0 and li == 0:
                    # Front legs grooming: sweep up toward head.
                    t = anim.groom_t
                    swing = -0.4 + 1.2 * abs(np.sin(t * np.pi))
                    lift = 0.35 * abs(np.sin(t * np.pi))
                    if side > 0:
                        swing *= 0.3
                p, c, i = _box(0.08, 0.45, 0.06, self.LEG)
                hip = root @ _mat_trs(0.05 + li * 0.08, 0.05 + lift, side * (0.2 + abs(z0)), pitch=0.9 + swing, roll=side * 0.4)
                _append_mesh(all_pos, all_col, all_idx, p, c, i, hip @ _mat_trs(0.0, -0.2, 0.0))

        positions = np.concatenate(all_pos, axis=0)
        colors = np.concatenate(all_col, axis=0)
        indices = np.concatenate(all_idx, axis=0)
        return positions, colors, indices


# Keep a module-level constant for gait phase offsets.
_LEG_PHASE_OFFSET = float(np.pi / 3.0)


class GltfFly(FlyModel):
    """Placeholder for a future GLTF/GLB fly asset swap."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"GLTF fly asset not found: {self.path}. "
                "Use ProceduralFly for PoC, or place a model at this path."
            )

    def mesh(self, pose: FlyPose, anim: AnimState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError(
            "GltfFly.load_gltf is a stub for later expansion. "
            f"Asset path reserved: {self.path}"
        )


def load_fly_model(path: Path | str | None = None) -> FlyModel:
    """Factory: procedural by default; GLTF when path is provided and exists."""
    if path is None:
        return ProceduralFly()
    p = Path(path)
    if p.exists():
        return GltfFly(p)
    return ProceduralFly()
