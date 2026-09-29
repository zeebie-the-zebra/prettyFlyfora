"""Ethogram → animation parameters for the procedural fly."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .pose import FlyPose


@dataclass
class AnimState:
    """Per-frame animation knobs consumed by ``ProceduralFly``."""

    wing_angle: float = 0.0  # radians, flap deflection
    wing_amp: float = 0.4
    leg_phase: float = 0.0  # gait cycle [0, 2π)
    leg_amp: float = 0.25
    groom_t: float = 0.0  # 0..1 grooming cycle progress
    limp: float = 0.0  # 0..1 stagger asymmetry
    lying: float = 0.0  # 0..1 collapsed blend


def update_anim(
    pose: FlyPose,
    *,
    time_s: float,
    prev: AnimState | None = None,
) -> AnimState:
    """Derive animation from behavior + locomotion speed."""
    prev = prev or AnimState()
    behavior = pose.behavior
    speed = max(0.0, pose.speed)

    if behavior == "collapsed":
        return AnimState(
            wing_angle=0.05,
            wing_amp=0.02,
            leg_phase=prev.leg_phase,
            leg_amp=0.0,
            groom_t=0.0,
            limp=0.0,
            lying=1.0,
        )

    if behavior == "pause":
        # Grooming: slow wing wipe + front-leg-over-head cycle.
        groom = (time_s * 0.7) % 1.0
        wing = 0.35 * math.sin(groom * math.pi * 2.0)
        return AnimState(
            wing_angle=wing,
            wing_amp=0.15,
            leg_phase=groom * math.tau,
            leg_amp=0.55,
            groom_t=groom,
            limp=0.0,
            lying=0.0,
        )

    limp = 0.55 if behavior == "stagger" else 0.0
    gait_rate = 6.0 + 8.0 * min(1.5, speed)
    if behavior == "stagger":
        gait_rate *= 0.45
    if behavior == "escape":
        gait_rate = 18.0

    flying = pose.in_flight or pose.y > 0.6 or behavior == "escape"
    wing_amp = 0.72 if flying else 0.22 + 0.2 * min(1.0, speed)
    if behavior == "stagger":
        wing_amp *= 0.4
    if behavior == "escape":
        wing_rate = 30.0
    elif flying:
        wing_rate = 18.0 + 6.0 * min(1.2, speed)
    else:
        wing_rate = 8.0 + 6.0 * speed

    return AnimState(
        wing_angle=wing_amp * math.sin(time_s * wing_rate),
        wing_amp=wing_amp,
        leg_phase=(time_s * gait_rate) % math.tau,
        leg_amp=0.05 if flying else (0.08 + 0.28 * min(1.0, speed)),
        groom_t=0.0,
        limp=limp,
        lying=0.0,
    )
