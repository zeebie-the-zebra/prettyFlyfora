"""Map WorldState → 3D presentation pose.

When ``enable_flight_state`` is on, altitude comes from sim state.
Otherwise altitude is a behavior-driven presentation lerp (legacy PoC).
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from flybrain.pixel_sim import WorldState


@dataclass
class FlyPose:
    """World-space pose for the fly mesh (Y-up)."""

    x: float
    y: float  # altitude
    z: float
    yaw: float  # radians, around Y
    pitch: float = 0.0
    roll: float = 0.0
    speed: float = 0.0
    behavior: str = "wander"
    in_flight: bool = False


# Presentation altitude targets when flight state is disabled.
_ALTITUDE = {
    "escape": 2.8,
    "desperate": 0.35,
    "stagger": 0.05,
    "collapsed": 0.0,
    "pause": 0.08,
    "search": 0.15,
    "approach": 0.2,
    "forage": 0.18,
    "upwind": 0.25,
    "wander": 0.12,
}


def from_state(
    state: WorldState,
    *,
    prev_altitude: float = 0.12,
    dt: float = 1.0 / 20.0,
    use_sim_altitude: bool = True,
) -> FlyPose:
    """Build pose from authoritative sim state.

    Sim ``fly_x`` / ``fly_y`` map to world X / Z.
    """
    behavior = state.behavior or "wander"

    if use_sim_altitude and hasattr(state, "altitude"):
        altitude = float(state.altitude)
        in_flight = bool(getattr(state, "in_flight", False))
    else:
        target_alt = float(_ALTITUDE.get(behavior, 0.12))
        target_alt += 0.15 * max(0.0, min(1.0, float(state.forward_drive) - 0.4))
        if behavior == "collapsed":
            target_alt = 0.0
        alpha = 1.0 - math.exp(-4.0 * max(dt, 1e-3))
        altitude = prev_altitude + (target_alt - prev_altitude) * alpha
        in_flight = altitude > 0.8

    pitch = 0.0
    roll = 0.0
    if behavior == "escape" or (in_flight and state.forward_drive > 0.9):
        pitch = -0.28
    elif behavior == "collapsed":
        pitch = 0.9
        roll = 0.4
    elif behavior == "stagger":
        roll = 0.15 * math.sin(state.tick * 0.4)
        pitch = -0.05
    elif in_flight:
        # Sustained cruise: slight nose-down, bank with turn.
        pitch = -0.14 - 0.06 * max(0.0, min(1.0, float(state.forward_drive)))
    elif altitude > 0.8:
        pitch = -0.12

    bank = 0.55 if in_flight else 0.35
    roll += bank * max(-1.0, min(1.0, float(state.turn_drive)))

    return FlyPose(
        x=float(state.fly_x),
        y=altitude,
        z=float(state.fly_y),
        yaw=float(state.heading),
        pitch=pitch,
        roll=roll,
        speed=float(state.forward_drive),
        behavior=behavior,
        in_flight=in_flight,
    )
