"""Third-person follow / orbit camera (Y-up)."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .pose import FlyPose


@dataclass
class FollowCamera:
    distance: float = 8.0
    height: float = 3.5
    yaw_offset: float = 0.0  # extra orbit yaw (radians)
    pitch: float = 0.35  # look-down angle
    smooth: float = 6.0
    min_distance: float = 3.0
    max_distance: float = 28.0

    eye: np.ndarray | None = None
    target: np.ndarray | None = None

    def reset_orbit(self) -> None:
        self.yaw_offset = 0.0
        self.distance = 8.0
        self.pitch = 0.35

    def orbit(self, dyaw: float, dpitch: float = 0.0) -> None:
        self.yaw_offset += dyaw
        self.pitch = float(np.clip(self.pitch + dpitch, 0.05, 1.2))

    def zoom(self, delta: float) -> None:
        self.distance = float(np.clip(self.distance * (1.0 + delta), self.min_distance, self.max_distance))

    def update(self, pose: FlyPose, dt: float) -> tuple[np.ndarray, np.ndarray]:
        target = np.array([pose.x, pose.y + 0.4, pose.z], dtype=np.float64)
        yaw = pose.yaw + math.pi + self.yaw_offset  # behind the fly
        # Spherical offset behind + above.
        horiz = self.distance * math.cos(self.pitch)
        eye_desired = np.array(
            [
                pose.x + horiz * math.cos(yaw),
                pose.y + self.height + self.distance * math.sin(self.pitch) * 0.35,
                pose.z + horiz * math.sin(yaw),
            ],
            dtype=np.float64,
        )

        if self.eye is None:
            self.eye = eye_desired.copy()
            self.target = target.copy()
        else:
            a = 1.0 - math.exp(-self.smooth * max(dt, 1e-4))
            self.eye = self.eye + (eye_desired - self.eye) * a
            self.target = self.target + (target - self.target) * a

        return self.eye, self.target


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Return a 4x4 view matrix (column-major friendly row layout for numpy @)."""
    if up is None:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / max(aspect, 1e-6)
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m
