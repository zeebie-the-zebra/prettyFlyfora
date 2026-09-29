"""Ground grid, food markers, loom threat disc."""

from __future__ import annotations

import numpy as np

from flybrain.pixel_sim import WorldState


def _line_grid(grid_size: int, y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Return line segment positions (Nx3) and colors for a ground grid."""
    positions = []
    colors = []
    col = (0.22, 0.18, 0.14)
    for i in range(grid_size + 1):
        # lines along X
        positions.append([0.0, y, float(i)])
        positions.append([float(grid_size), y, float(i)])
        colors.extend([col, col])
        # lines along Z
        positions.append([float(i), y, 0.0])
        positions.append([float(i), y, float(grid_size)])
        colors.extend([col, col])
    return np.asarray(positions, dtype=np.float32), np.asarray(colors, dtype=np.float32)


def ground_plane(grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subtle ground quad under the grid."""
    s = float(grid_size)
    positions = np.array(
        [
            [0.0, -0.02, 0.0],
            [s, -0.02, 0.0],
            [s, -0.02, s],
            [0.0, -0.02, s],
        ],
        dtype=np.float32,
    )
    colors = np.tile(np.array([0.14, 0.12, 0.10], dtype=np.float32), (4, 1))
    indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    return positions, colors, indices


def food_markers(state: WorldState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small cubes at food cells."""
    if not state.foods:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.uint32),
        )
    all_pos = []
    all_col = []
    all_idx = []
    color = (0.86, 0.35, 0.24)
    for fx, fy in state.foods:
        hx = 0.35
        base = len(all_pos)
        corners = [
            [fx + 0.5 - hx, 0.0, fy + 0.5 - hx],
            [fx + 0.5 + hx, 0.0, fy + 0.5 - hx],
            [fx + 0.5 + hx, 0.7, fy + 0.5 - hx],
            [fx + 0.5 - hx, 0.7, fy + 0.5 - hx],
            [fx + 0.5 - hx, 0.0, fy + 0.5 + hx],
            [fx + 0.5 + hx, 0.0, fy + 0.5 + hx],
            [fx + 0.5 + hx, 0.7, fy + 0.5 + hx],
            [fx + 0.5 - hx, 0.7, fy + 0.5 + hx],
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (0, 3, 7, 4),
            (1, 2, 6, 5),
        ]
        for face in faces:
            b = len(all_pos)
            for vi in face:
                all_pos.append(corners[vi])
                all_col.append(color)
            all_idx.extend([b, b + 1, b + 2, b, b + 2, b + 3])
    return (
        np.asarray(all_pos, dtype=np.float32),
        np.asarray(all_col, dtype=np.float32),
        np.asarray(all_idx, dtype=np.uint32),
    )


def threat_disc(state: WorldState, segments: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Line-loop circle for looming threat (empty if inactive)."""
    if state.threat_x is None or state.threat_y is None or state.threat_radius <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    cx, cz = float(state.threat_x), float(state.threat_y)
    r = float(state.threat_radius) * 0.35
    positions = []
    colors = []
    col = (0.7, 0.2, 0.15)
    for i in range(segments + 1):
        a = 2 * np.pi * i / segments
        positions.append([cx + r * np.cos(a), 0.05, cz + r * np.sin(a)])
        colors.append(col)
    # duplicate for GL_LINES strip pairs
    line_pos = []
    line_col = []
    for i in range(segments):
        line_pos.append(positions[i])
        line_pos.append(positions[i + 1])
        line_col.append(col)
        line_col.append(col)
    return np.asarray(line_pos, dtype=np.float32), np.asarray(line_col, dtype=np.float32)


def edge_walls(grid_size: int, height: float = 2.5, thickness: float = 0.18) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four perimeter walls around the sandbox (inward faces)."""
    s = float(grid_size)
    t = thickness
    h = height
    color = (0.28, 0.24, 0.20)
    # (cx, cy, cz, sx, sy, sz) for each wall slab
    specs = [
        (s * 0.5, h * 0.5, -t * 0.5, s + 2 * t, h, t),  # south (z=0)
        (s * 0.5, h * 0.5, s + t * 0.5, s + 2 * t, h, t),  # north (z=s)
        (-t * 0.5, h * 0.5, s * 0.5, t, h, s),  # west (x=0)
        (s + t * 0.5, h * 0.5, s * 0.5, t, h, s),  # east (x=s)
    ]
    all_pos: list = []
    all_col: list = []
    all_idx: list = []
    for cx, cy, cz, sx, sy, sz in specs:
        hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
        corners = [
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (2, 3, 7, 6),
            (0, 3, 7, 4),
            (1, 2, 6, 5),
        ]
        for face in faces:
            b = len(all_pos)
            for vi in face:
                all_pos.append(corners[vi])
                all_col.append(color)
            all_idx.extend([b, b + 1, b + 2, b, b + 2, b + 3])
    return (
        np.asarray(all_pos, dtype=np.float32),
        np.asarray(all_col, dtype=np.float32),
        np.asarray(all_idx, dtype=np.uint32),
    )


def build_static_world(grid_size: int) -> dict:
    gp, gc, gi = ground_plane(grid_size)
    lp, lc = _line_grid(grid_size)
    wp, wc, wi = edge_walls(grid_size)
    return {
        "ground_pos": gp,
        "ground_col": gc,
        "ground_idx": gi,
        "grid_pos": lp,
        "grid_col": lc,
        "wall_pos": wp,
        "wall_col": wc,
        "wall_idx": wi,
    }
