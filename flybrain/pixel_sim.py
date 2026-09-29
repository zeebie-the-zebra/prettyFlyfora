"""
Phase 3 — multi-modal fly ethogram on the Phase-1 v2 connectome.

World senses → SENSORY modalities:
  visual        nearest-food bearing (photoreceptors)
  olfactory     odor plume field from food (concentration + local gradient)
  mechano_wind  global wind relative to heading
  gustatory     contact taste when on/near food
  visual_loom   LPLC2 loom flashes (escape)

Behaviours: wander | forage | approach | pause | escape | search | upwind
           | desperate | stagger | collapsed  (starvation arc)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import random
from typing import Sequence

import numpy as np

from pathlib import Path

from .io import DEFAULT_PROCESSED, load_network
from .model import RateNetwork
from .motor_decode import DecodeResult, MotorDecoder, load_motor_calibration


@dataclass
class SimConfig:
    grid_size: int = 64
    n_food: int = 8
    food_respawn: bool = True
    food_respawn_delay: int = 50
    neural_steps_per_tick: int = 8
    sensory_rate: float = 1.0
    # Multiplies all computed sensory modality drives (visual/olfactory/wind/taste/loom)
    # before the connectome clamps them into SENSOR neurons.
    sensory_gain_mult: float = 1.0
    sensory_clamp_max: float = 2.5
    max_sense_dist: float = 40.0
    move_speed: float = 1.0
    turn_rate: float = 0.45
    relative_motor_decode: bool = True
    idle_forward: float = 0.28
    seed: int = 0
    controller: str = "connectome"

    # Hunger
    energy_start: float = 0.80
    # ~1 hour from full energy at default 20 FPS with no food (was 1/400 ≈ 20s).
    energy_drain_per_tick: float = 1.0 / (20.0 * 3600.0)
    # Stage A defaults: earlier hunger, smaller meals, less seek-assist scaffolding.
    energy_per_food: float = 0.40
    seek_hunger_start: float = 0.20
    seek_hunger_full: float = 0.80
    sated_sense_floor: float = 0.05
    connectome_seek_assist: float = 0.25
    eat_min_hunger: float = 0.25
    forage_motivation_threshold: float = 0.15
    # Default wander blend: wander_weight * wander + (1-wander_weight) * brain
    wander_weight: float = 0.65

    # Starvation arc once energy hits 0:
    #   desperate forage → stagger → collapse → auto-respawn
    starve_desperate_ticks: int = 90   # ~4.5s @20fps — frantic last search
    starve_stagger_ticks: int = 60     # ~3s — slowing / failing
    collapse_respawn_ticks: int = 80   # lie still, then soft respawn
    stagger_speed_scale: float = 0.22
    stagger_turn_scale: float = 0.55

    # Odor plume (each food is a source)
    odor_sigma: float = 10.0
    odor_strength: float = 1.2
    odor_antenna_offset: float = 1.2  # sample L/R of heading
    # Stretch plumes downwind (1 = isotropic; >1 elongates along wind).
    wind_odor_stretch: float = 1.8

    # Wind
    wind_speed: float = 0.55
    wind_angle: float = 0.0  # world radians; 0 = +x
    wind_meander: float = 0.02  # slow random walk of wind angle

    # Near-wall mechanosense injected into mechano_wind (walled mode).
    wall_mechano_gain: float = 0.85
    wall_mechano_range: float = 5.0
    # Residual post-hoc turn bias (on top of sensory wall mechano).
    wall_avoid_gain: float = 0.45

    # Optic-flow lite: yaw/forward motion into visual drive.
    optic_flow_gain: float = 0.35

    # Taste
    taste_radius: float = 1.6

    # Stop-and-go
    pause_prob_per_tick: float = 0.012
    pause_min_ticks: int = 8
    pause_max_ticks: int = 28
    pause_suppress_when_hunger: float = 0.85

    # Loom / escape
    loom_prob_per_tick: float = 0.0035
    loom_duration: int = 12
    loom_sensory: float = 1.4
    escape_turn: float = 1.2
    escape_forward: float = 1.6

    # Post-meal cast + upwind preference when hungry in odor
    search_duration: int = 40
    search_turn_gain: float = 0.9
    upwind_odor_threshold: float = 0.15
    odor_loss_cast_threshold: float = 0.04

    approach_cos_threshold: float = 0.75
    approach_max_dist: float = 12.0

    # Pause / groom: emergent gate on top of stochastic pauses
    pause_low_drive: float = 0.22
    pause_low_odor_grad: float = 0.04
    pause_emergent_prob: float = 0.04

    # Flight mode (Stage D — real walk/fly locomotion, not presentation-only).
    enable_flight_state: bool = True
    flight_takeoff_altitude: float = 2.4
    flight_cruise_altitude: float = 1.65
    flight_land_altitude: float = 0.12
    flight_escape_boost: float = 1.35
    # Sustained cruise locomotion (applied while in_flight, non-escape).
    flight_cruise_speed_mult: float = 1.65
    flight_turn_scale: float = 0.70
    flight_energy_mult: float = 2.2
    # Voluntary takeoff when walking with enough drive / energy / clearance.
    flight_voluntary_prob: float = 0.028
    flight_takeoff_drive: float = 0.45
    flight_takeoff_min_energy: float = 0.28
    flight_min_food_dist: float = 4.0
    flight_land_food_dist: float = 2.8
    flight_max_ticks: int = 360
    flight_land_prob: float = 0.008
    flight_cooldown_ticks: int = 40
    # Extra takeoff chance when hungry in open plume (aerial search).
    flight_forage_boost: float = 0.035

    network: str = "reduced"
    processed_dir: Path = DEFAULT_PROCESSED
    motor_panel_max: int = 24
    # True: toroidal wrap (classic 2D). False: bounce off sandbox walls (3D view).
    wrap_world: bool = True
    wall_margin: float = 0.65
    note_left_skew: str = (
        "Phase-1 v2 multi-modal subgraph; mild L/R asymmetries can remain."
    )


@dataclass
class FoodEvent:
    tick: int
    x: int
    y: int


@dataclass
class WorldState:
    tick: int = 0
    fly_x: float = 32.0
    fly_y: float = 32.0
    heading: float = 0.0
    energy: float = 0.75
    hunger: float = 0.25
    motivation: float = 0.0
    foods: list[tuple[int, int]] = field(default_factory=list)
    pending_respawns: list[int] = field(default_factory=list)
    foods_eaten: list[FoodEvent] = field(default_factory=list)
    motor_rates: np.ndarray = field(default_factory=lambda: np.zeros(0))
    motor_decode: DecodeResult | None = None
    # Mean modality drives (for UI)
    sense: dict[str, float] = field(default_factory=dict)
    turn_drive: float = 0.0
    forward_drive: float = 0.0
    # Connectome-path diagnostics (pre/post blend)
    sensory_gain_used: float = 1.0
    sensory_clamp_used: float = 2.5
    sensory_pre_clamp_max: float = 0.0
    sensory_post_clamp_max: float = 0.0
    conn_turn: float = 0.0
    conn_forward: float = 0.0
    seek_turn: float = 0.0
    brain_turn: float = 0.0
    wander_turn: float = 0.0
    wander_forward: float = 0.0
    connectome_seek_assist_used: float = 0.0
    behavior: str = "wander"
    wind_angle: float = 0.0
    odor: float = 0.0
    loom_timer: int = 0
    loom_side: str = "both"
    pause_timer: int = 0
    search_timer: int = 0
    threat_x: float | None = None
    threat_y: float | None = None
    threat_radius: float = 0.0
    # Starvation
    empty_ticks: int = 0  # consecutive ticks at energy == 0
    starvation_stage: str = "ok"  # ok | desperate | stagger | collapsed
    collapse_timer: int = 0
    deaths: int = 0
    # Flight (Stage D sim locomotion: walk vs sustained flight)
    in_flight: bool = False
    altitude: float = 0.12
    flight_ticks: int = 0
    flight_cooldown: int = 0
    locomotion: str = "walk"  # walk | flight


class Controller(str, Enum):
    CONNECTOME = "connectome"
    GREEDY = "greedy"
    RANDOM = "random"


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 0.0 if x < edge0 else 1.0
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _nearest_food(
    fx: float, fy: float, foods: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int] | None, float, float]:
    if not foods:
        return None, float("inf"), 0.0
    best = None
    best_d = float("inf")
    best_bearing = 0.0
    for x, y in foods:
        dx = x + 0.5 - fx
        dy = y + 0.5 - fy
        d = math.hypot(dx, dy)
        if d < best_d:
            best_d = d
            best = (x, y)
            best_bearing = math.atan2(dy, dx)
    return best, best_d, best_bearing


class PixelSim:
    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()
        self.rng = random.Random(self.config.seed)
        self.np_rng = np.random.default_rng(self.config.seed)

        cfg = self.config
        network = cfg.network
        if network not in ("reduced", "full"):
            raise ValueError(f"network must be 'reduced' or 'full', got {network!r}")

        self.net = load_network(cfg.processed_dir, network=network)
        self.brain = RateNetwork(self.net, leak=0.85, gain=0.8)
        self.motor_cal = load_motor_calibration(
            self.net, network=network, processed_dir=cfg.processed_dir
        )
        self.motor_decoder = MotorDecoder(
            self.motor_cal,
            relative=cfg.relative_motor_decode,
            idle_forward=cfg.idle_forward,
        )
        meta = self.net.neurons.iloc[self.net.motor_idx].reset_index(drop=True)
        self.motor_sides = meta["side"].to_numpy()
        self.motor_types = meta["primary_type"].astype(str).to_numpy()
        self.motor_labels = [f"{t}_{s}" for t, s in zip(self.motor_types, self.motor_sides)]
        self.motor_roles = self.motor_cal.roles
        self._neural_store = "motor_only" if network == "full" else "all"
        self._sensory_config_verified = False
        self._prev_heading = 0.0
        self._prev_odor = 0.0

        print(
            f"[PixelSim] network={network}  sensory_gain_mult={cfg.sensory_gain_mult}  "
            f"sensory_clamp_max={cfg.sensory_clamp_max}  neural_store={self._neural_store}"
        )

        self.state = WorldState()
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self.state = WorldState(
            fly_x=cfg.grid_size / 2,
            fly_y=cfg.grid_size / 2,
            heading=self.rng.uniform(0, 2 * math.pi),
            energy=cfg.energy_start,
            hunger=1.0 - cfg.energy_start,
            foods=[],
            motor_rates=np.zeros(len(self.motor_labels), dtype=np.float64),
            wind_angle=cfg.wind_angle,
            sense={},
            in_flight=False,
            altitude=cfg.flight_land_altitude,
            flight_ticks=0,
            flight_cooldown=0,
            locomotion="walk",
        )
        for _ in range(cfg.n_food):
            self._spawn_food()
        self.brain.reset()
        self._prev_heading = self.state.heading
        self._prev_odor = 0.0

    def _spawn_food(self) -> None:
        g = self.config.grid_size
        occupied = set(self.state.foods)
        for _ in range(1000):
            x = self.rng.randrange(g)
            y = self.rng.randrange(g)
            if (x, y) in occupied:
                continue
            if int(self.state.fly_x) == x and int(self.state.fly_y) == y:
                continue
            self.state.foods.append((x, y))
            return

    def trigger_loom(self, side: str | None = None) -> None:
        cfg = self.config
        st = self.state
        st.loom_timer = cfg.loom_duration
        st.loom_side = side or self.rng.choice(["left", "right", "both"])
        st.pause_timer = 0
        ang = st.heading + self.rng.uniform(-0.6, 0.6)
        tx = st.fly_x + 10 * math.cos(ang)
        ty = st.fly_y + 10 * math.sin(ang)
        if cfg.wrap_world:
            tx %= cfg.grid_size
            ty %= cfg.grid_size
        else:
            tx = min(cfg.grid_size - 0.5, max(0.5, tx))
            ty = min(cfg.grid_size - 0.5, max(0.5, ty))
        st.threat_x = tx
        st.threat_y = ty
        st.threat_radius = 1.0

    # --- environment physics -------------------------------------------------

    def _displacement(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float]:
        """Vector from (x0,y0) to (x1,y1); toroidal only when wrap_world is on."""
        dx = x1 - x0
        dy = y1 - y0
        if self.config.wrap_world:
            g = self.config.grid_size
            if abs(dx) > g / 2:
                dx -= math.copysign(g, dx)
            if abs(dy) > g / 2:
                dy -= math.copysign(g, dy)
        return dx, dy

    def _odor_at(self, x: float, y: float) -> float:
        """Odor concentration; plumes elongate downwind when wind_odor_stretch > 1."""
        cfg = self.config
        total = 0.0
        wx = math.cos(self.state.wind_angle)
        wy = math.sin(self.state.wind_angle)
        stretch = max(1.0, float(cfg.wind_odor_stretch))
        sigma_across = cfg.odor_sigma * 0.65
        sigma_along = cfg.odor_sigma * stretch
        for fx, fy in self.state.foods:
            # Vector from source → sample point
            dx, dy = self._displacement(fx + 0.5, fy + 0.5, x, y)
            along = dx * wx + dy * wy  # + downwind of food
            across = -dx * wy + dy * wx
            # Weak upwind lobe so plume isn't only downwind.
            upwind_gate = 0.2 + 0.8 * (0.5 + 0.5 * math.tanh(along / max(cfg.odor_sigma, 1e-3)))
            r2 = (along / sigma_along) ** 2 + (across / sigma_across) ** 2
            total += cfg.odor_strength * upwind_gate * math.exp(-0.5 * r2)
        return total

    def _wall_avoid_turn(self) -> float:
        """Steer toward map interior when near walls (walled sandbox only)."""
        cfg = self.config
        if cfg.wrap_world:
            return 0.0
        st = self.state
        g = float(cfg.grid_size)
        dist_edge = min(st.fly_x, st.fly_y, g - st.fly_x, g - st.fly_y)
        sense = 6.0
        if dist_edge >= sense:
            return 0.0
        desired = math.atan2(g * 0.5 - st.fly_y, g * 0.5 - st.fly_x)
        err = _wrap_angle(desired - st.heading)
        strength = (sense - dist_edge) / sense
        return max(-1.2, min(1.2, 1.8 * err * strength * cfg.wall_avoid_gain))

    def _wall_mechano_lr(self) -> tuple[float, float]:
        """Near-wall contact → left/right mechanosensory drive (walled mode)."""
        cfg = self.config
        if cfg.wrap_world:
            return 0.0, 0.0
        st = self.state
        g = float(cfg.grid_size)
        sense = cfg.wall_mechano_range
        gain = cfg.wall_mechano_gain
        left = right = 0.0
        # Wall normals point inward; bearing is direction toward the wall exterior.
        walls = (
            (math.pi, st.fly_x),  # west wall (x=0), outside is -X
            (0.0, g - st.fly_x),  # east
            (-math.pi / 2, st.fly_y),  # south (y=0)
            (math.pi / 2, g - st.fly_y),  # north
        )
        for bearing, dist in walls:
            if dist >= sense:
                continue
            amp = gain * (1.0 - dist / sense)
            rel = _wrap_angle(bearing - st.heading)
            left += amp * max(0.0, 0.5 + 0.5 * math.sin(rel))
            right += amp * max(0.0, 0.5 - 0.5 * math.sin(rel))
        return left, right

    def _update_wind(self) -> None:
        cfg = self.config
        st = self.state
        st.wind_angle = _wrap_angle(
            st.wind_angle + self.rng.uniform(-cfg.wind_meander, cfg.wind_meander)
        )

    def _sense_all(self, motivation: float) -> dict[str, tuple[float, float]]:
        """Return modality → (left, right) drives in ~[0, 1.5]."""
        cfg = self.config
        st = self.state
        gate = cfg.sated_sense_floor + (1.0 - cfg.sated_sense_floor) * motivation

        # Visual: food bearing + optic-flow lite from yaw / forward motion
        food, dist, world_bearing = _nearest_food(st.fly_x, st.fly_y, st.foods)
        if food is None or dist > cfg.max_sense_dist:
            vis_l = vis_r = 0.0
            rel = 0.0
        else:
            rel = _wrap_angle(world_bearing - st.heading)
            strength = cfg.sensory_rate * max(0.0, 1.0 - dist / cfg.max_sense_dist)
            left_w = 0.5 + 0.5 * math.sin(rel)
            right_w = 0.5 - 0.5 * math.sin(rel)
            ahead = 0.55 + 0.45 * max(0.0, math.cos(rel))
            vis_l = strength * left_w * ahead * gate
            vis_r = strength * right_w * ahead * gate
        # Optic flow: yaw rate → contralateral bias; forward → mild bilateral.
        d_yaw = _wrap_angle(st.heading - self._prev_heading)
        ofg = cfg.optic_flow_gain
        yaw_flow = ofg * min(1.5, abs(d_yaw) * 4.0)
        # Turning left (d_yaw>0 in CCW) → more right-eye motion.
        if d_yaw >= 0:
            vis_r += yaw_flow
            vis_l += 0.35 * yaw_flow
        else:
            vis_l += yaw_flow
            vis_r += 0.35 * yaw_flow
        fwd_flow = ofg * 0.12 * max(0.0, float(st.forward_drive))
        vis_l += fwd_flow
        vis_r += fwd_flow
        st._food_rel = rel  # type: ignore[attr-defined]
        st._food_dist = dist  # type: ignore[attr-defined]

        # Olfactory: concentration + antennae gradient
        off = cfg.odor_antenna_offset
        lx = st.fly_x + off * math.cos(st.heading + math.pi / 2)
        ly = st.fly_y + off * math.sin(st.heading + math.pi / 2)
        rx = st.fly_x + off * math.cos(st.heading - math.pi / 2)
        ry = st.fly_y + off * math.sin(st.heading - math.pi / 2)
        if cfg.wrap_world:
            odor_l = self._odor_at(lx % cfg.grid_size, ly % cfg.grid_size)
            odor_r = self._odor_at(rx % cfg.grid_size, ry % cfg.grid_size)
            odor_c = self._odor_at(st.fly_x, st.fly_y)
        else:
            # Outside the sandbox: no odor (walls block wrap-around smell).
            def _odor_in_bounds(x: float, y: float) -> float:
                if x < 0 or y < 0 or x >= cfg.grid_size or y >= cfg.grid_size:
                    return 0.0
                return self._odor_at(x, y)

            odor_l = _odor_in_bounds(lx, ly)
            odor_r = _odor_in_bounds(rx, ry)
            odor_c = self._odor_at(st.fly_x, st.fly_y)
        st.odor = odor_c
        # Hunger gates odor seeking too (sated flies ignore plumes).
        olf_l = min(1.5, odor_l) * gate
        olf_r = min(1.5, odor_r) * gate

        # Wind: ipsilateral mechanosensory from wind relative to heading
        wind_rel = _wrap_angle(st.wind_angle - st.heading)
        # Headwind (wind from front) → both; from left → left antennae more.
        speed = cfg.wind_speed
        wind_l = speed * (0.45 + 0.55 * max(0.0, math.sin(wind_rel)))
        wind_r = speed * (0.45 + 0.55 * max(0.0, -math.sin(wind_rel)))
        # Mild headwind boost
        head = max(0.0, math.cos(wind_rel))
        wind_l *= 0.7 + 0.3 * head
        wind_r *= 0.7 + 0.3 * head
        # Near-wall mechanosense (walled sandbox) — real sensory path, not only post-hoc turn.
        wl, wr = self._wall_mechano_lr()
        wind_l += wl
        wind_r += wr
        wind_l = min(2.0, wind_l)
        wind_r = min(2.0, wind_r)

        # Gustatory: contact
        taste = 0.0
        if dist <= cfg.taste_radius:
            taste = 1.0 * max(gate, 0.3)  # even somewhat full flies taste contact
        gust_l = gust_r = taste

        # Loom → LPLC2 (ungated)
        loom_l = loom_r = 0.0
        if st.loom_timer > 0:
            progress = 1.0 - (st.loom_timer / cfg.loom_duration)
            st.threat_radius = 1.0 + 8.0 * progress
            flash = cfg.loom_sensory * (0.6 + 0.4 * progress)
            if st.loom_side in ("left", "both"):
                loom_l = flash
            if st.loom_side in ("right", "both"):
                loom_r = flash

        mult = float(cfg.sensory_gain_mult)
        return {
            "visual": (vis_l * mult, vis_r * mult),
            "olfactory": (olf_l * mult, olf_r * mult),
            "mechano_wind": (wind_l * mult, wind_r * mult),
            "gustatory": (gust_l * mult, gust_r * mult),
            "visual_loom": (loom_l * mult, loom_r * mult),
        }

    def _update_flight_state(self, behavior: str) -> None:
        """Stage D walk/fly: voluntary cruise + escape takeoff + food/energy landings."""
        cfg = self.config
        st = self.state
        if not cfg.enable_flight_state:
            st.in_flight = False
            st.altitude = cfg.flight_land_altitude
            st.flight_ticks = 0
            st.locomotion = "walk"
            return

        if st.flight_cooldown > 0:
            st.flight_cooldown -= 1

        food_dist = float(getattr(st, "_food_dist", 99.0))
        was_flying = bool(st.in_flight)

        # --- Forced land ---
        force_land = (
            behavior in ("collapsed", "stagger", "pause", "approach")
            or st.energy < 0.10
            or food_dist < cfg.flight_land_food_dist
        )
        # Soft land: long flight, or random while cruising (stronger when hungry near odor).
        soft_land = False
        if st.in_flight and not force_land and behavior != "escape":
            if st.flight_ticks >= cfg.flight_max_ticks:
                soft_land = True
            elif self.rng.random() < cfg.flight_land_prob:
                soft_land = True
            elif st.hunger > 0.55 and st.odor > 0.2 and food_dist < cfg.approach_max_dist:
                soft_land = self.rng.random() < 0.08

        # --- Takeoff ---
        takeoff = False
        if behavior == "escape":
            takeoff = True
        elif (
            not st.in_flight
            and not force_land
            and st.flight_cooldown <= 0
            and st.energy >= cfg.flight_takeoff_min_energy
            and food_dist >= cfg.flight_min_food_dist
            and behavior
            in ("wander", "forage", "upwind", "search", "desperate")
        ):
            drive_ok = float(st.forward_drive) >= cfg.flight_takeoff_drive
            p = cfg.flight_voluntary_prob
            if behavior in ("forage", "upwind", "desperate") and st.hunger > 0.35:
                p += cfg.flight_forage_boost
            if drive_ok and self.rng.random() < p:
                takeoff = True
            # Strong sustained push → launch even without random roll.
            elif float(st.forward_drive) >= cfg.flight_takeoff_drive + 0.25 and self.rng.random() < 0.08:
                takeoff = True

        if force_land or soft_land:
            st.in_flight = False
        elif takeoff:
            st.in_flight = True

        # Altitude targets
        if behavior == "escape" and st.in_flight:
            target = cfg.flight_takeoff_altitude
        elif behavior == "collapsed":
            target = 0.0
        elif st.in_flight:
            target = cfg.flight_cruise_altitude
            # Slight climb with speed, dip when turning hard.
            target += 0.25 * max(0.0, min(1.0, float(st.forward_drive) - 0.4))
            target -= 0.15 * min(1.0, abs(float(st.turn_drive)))
        else:
            target = cfg.flight_land_altitude

        alpha = 0.28 if st.in_flight or behavior == "escape" else 0.55
        st.altitude = float(st.altitude + (target - st.altitude) * alpha)

        if st.in_flight:
            st.flight_ticks += 1
            st.locomotion = "flight"
        else:
            if was_flying:
                st.flight_cooldown = cfg.flight_cooldown_ticks
            st.flight_ticks = 0
            st.locomotion = "walk"

    def _apply_flight_locomotion(
        self, turn: float, forward: float, behavior: str
    ) -> tuple[float, float]:
        """Aerial speed/turn profile while cruising (escape keeps its own boost)."""
        cfg = self.config
        st = self.state
        if not cfg.enable_flight_state or not st.in_flight:
            return turn, forward
        if behavior == "escape":
            return turn, forward
        if behavior in ("pause", "collapsed"):
            return turn, 0.0
        forward = max(0.0, forward) * cfg.flight_cruise_speed_mult
        turn = turn * cfg.flight_turn_scale
        # Keep a minimum cruise airspeed so flight doesn't stall into a float.
        forward = max(forward, 0.35 * cfg.flight_cruise_speed_mult)
        return turn, forward

    def _decode_motor(self, motor_rates: np.ndarray) -> DecodeResult:
        return self.motor_decoder.decode(motor_rates)

    def _wander(self) -> tuple[float, float]:
        return self.rng.uniform(-0.35, 0.35), self.config.idle_forward

    def _soft_respawn(self) -> None:
        """After collapse: revive fly in-place (keep world / score)."""
        cfg = self.config
        st = self.state
        st.energy = cfg.energy_start
        st.hunger = 1.0 - cfg.energy_start
        st.motivation = 0.0
        st.empty_ticks = 0
        st.starvation_stage = "ok"
        st.collapse_timer = 0
        st.pause_timer = 0
        st.search_timer = 0
        st.loom_timer = 0
        st.threat_x = st.threat_y = None
        st.threat_radius = 0.0
        st.heading = self.rng.uniform(0, 2 * math.pi)
        st.in_flight = False
        st.altitude = self.config.flight_land_altitude
        st.flight_ticks = 0
        st.flight_cooldown = 0
        st.locomotion = "walk"
        # Nudge away from current cell so we don't instantly re-eat / re-starve vibes.
        st.fly_x = (st.fly_x + self.rng.uniform(-6, 6)) % cfg.grid_size
        st.fly_y = (st.fly_y + self.rng.uniform(-6, 6)) % cfg.grid_size
        self.brain.reset()

    def _update_starvation(self) -> str:
        """Advance empty-energy timers; return stage ok|desperate|stagger|collapsed."""
        cfg = self.config
        st = self.state

        if st.starvation_stage == "collapsed":
            st.collapse_timer -= 1
            if st.collapse_timer <= 0:
                self._soft_respawn()
                return "ok"
            return "collapsed"

        if st.energy > 1e-9:
            st.empty_ticks = 0
            st.starvation_stage = "ok"
            return "ok"

        # Energy is zero.
        st.empty_ticks += 1
        if st.empty_ticks <= cfg.starve_desperate_ticks:
            st.starvation_stage = "desperate"
        elif st.empty_ticks <= cfg.starve_desperate_ticks + cfg.starve_stagger_ticks:
            st.starvation_stage = "stagger"
        else:
            st.starvation_stage = "collapsed"
            st.collapse_timer = cfg.collapse_respawn_ticks
            st.deaths += 1
        return st.starvation_stage

    def _choose_behavior(self, motivation: float, drives: dict) -> str:
        cfg = self.config
        st = self.state
        if st.starvation_stage == "collapsed":
            return "collapsed"
        if st.starvation_stage == "stagger":
            return "stagger"
        if st.loom_timer > 0:
            return "escape"
        if st.starvation_stage == "desperate":
            # Last-chance food search — skip casual pauses.
            rel = getattr(st, "_food_rel", 0.0)
            dist = getattr(st, "_food_dist", 99.0)
            if dist < cfg.approach_max_dist and math.cos(rel) > cfg.approach_cos_threshold:
                return "approach"
            if st.odor > cfg.upwind_odor_threshold:
                return "upwind"
            return "desperate"
        if st.pause_timer > 0:
            return "pause"
        if st.search_timer > 0:
            return "search"
        rel = getattr(st, "_food_rel", 0.0)
        dist = getattr(st, "_food_dist", 99.0)
        if (
            motivation > 0.2
            and dist < cfg.approach_max_dist
            and math.cos(rel) > cfg.approach_cos_threshold
        ):
            return "approach"
        if motivation > 0.25 and st.odor > cfg.upwind_odor_threshold:
            return "upwind"
        if motivation > cfg.forage_motivation_threshold:
            return "forage"
        return "wander"

    def _connectome_action(
        self, drives: dict, motivation: float, behavior: str
    ) -> tuple[float, float]:
        cfg = self.config
        st = self.state
        gain = float(cfg.sensory_gain_mult)
        clamp = float(cfg.sensory_clamp_max)
        pre_clamp_max = max(
            (max(abs(v[0]), abs(v[1])) for v in drives.values()),
            default=0.0,
        )
        if not self._sensory_config_verified:
            assert gain == cfg.sensory_gain_mult, "sensory_gain_mult mismatch in connectome path"
            assert clamp == cfg.sensory_clamp_max, "sensory_clamp_max mismatch in connectome path"
            print(
                f"[PixelSim] sensory config verified on first connectome tick: "
                f"network={cfg.network} gain={gain} clamp={clamp} "
                f"pre_clamp_max={pre_clamp_max:.4f}"
            )
            self._sensory_config_verified = True

        self.brain.set_sensory_modalities(drives, clamp_max=clamp)
        post_clamp_max = float(np.max(self.brain._sensory_rates)) if self.brain._sensory_rates.size else 0.0
        st.sensory_gain_used = gain
        st.sensory_clamp_used = clamp
        st.sensory_pre_clamp_max = pre_clamp_max
        st.sensory_post_clamp_max = post_clamp_max

        out = self.brain.run(cfg.neural_steps_per_tick, store=self._neural_store)
        st.motor_rates = out["motor"][-1].copy()
        decoded = self._decode_motor(st.motor_rates)
        st.motor_decode = decoded
        conn_turn, conn_fwd = decoded.turn, decoded.forward
        st.conn_turn = conn_turn
        st.conn_forward = conn_fwd

        vis_l, vis_r = drives.get("visual", (0.0, 0.0))
        olf_l, olf_r = drives.get("olfactory", (0.0, 0.0))
        # Multimodal taxis assist: vision + odor gradient
        seek_turn = 0.55 * (vis_l - vis_r) + 0.45 * (olf_l - olf_r)
        st.seek_turn = seek_turn
        assist = cfg.connectome_seek_assist * motivation
        st.connectome_seek_assist_used = assist
        brain_turn = (1.0 - assist) * conn_turn + assist * seek_turn
        st.brain_turn = brain_turn
        wander_turn, wander_fwd = self._wander()
        st.wander_turn = wander_turn
        st.wander_forward = wander_fwd

        if behavior == "collapsed":
            return 0.0, 0.0

        if behavior == "stagger":
            wobble = self.rng.uniform(-1.0, 1.0)
            turn = cfg.stagger_turn_scale * (0.4 * brain_turn + 0.6 * wobble)
            forward = cfg.stagger_speed_scale * max(0.15, conn_fwd)
            return turn, forward

        if behavior == "desperate":
            turn = 0.25 * brain_turn + 0.75 * seek_turn
            if abs(seek_turn) < 0.05:
                turn += self.rng.uniform(-0.8, 0.8)
            forward = max(0.75, conn_fwd)
            return turn, forward

        if behavior == "escape":
            if self.state.loom_side == "left":
                turn = -cfg.escape_turn
            elif self.state.loom_side == "right":
                turn = cfg.escape_turn
            else:
                turn = self.rng.choice([-1.0, 1.0]) * cfg.escape_turn
            fwd = max(cfg.escape_forward, conn_fwd)
            if cfg.enable_flight_state and self.state.in_flight:
                fwd *= cfg.flight_escape_boost
            return 0.5 * turn + 0.5 * brain_turn, fwd

        if behavior == "pause":
            return brain_turn * 0.1, 0.0

        if behavior == "search":
            # Odor-loss cast: stronger zig-zag when plume concentration falls.
            odor_drop = float(self._prev_odor) - float(self.state.odor)
            cast = cfg.search_turn_gain * math.sin(self.state.tick / 4.0)
            if odor_drop > cfg.odor_loss_cast_threshold:
                turn = 0.2 * brain_turn + 0.8 * cast
                forward = 0.18
            elif self.state.odor < 0.08:
                turn = 0.35 * brain_turn + 0.65 * cast
                forward = 0.22
            else:
                turn = 0.6 * brain_turn + 0.4 * cast * 0.5
                forward = max(0.25, min(0.45, conn_fwd))
            return turn, forward

        if behavior == "approach":
            return 0.3 * brain_turn + 0.7 * seek_turn, min(0.5, max(0.2, conn_fwd * 0.7))

        if behavior == "upwind":
            # Turn toward wind source (upwind = into the wind).
            wind_rel = _wrap_angle(self.state.wind_angle - self.state.heading)
            # Face into wind: want wind_rel → ±π (headwind). Simpler: align heading to wind_angle+π
            desired = _wrap_angle(self.state.wind_angle + math.pi - self.state.heading)
            upwind_turn = max(-1.0, min(1.0, desired * 1.5))
            turn = 0.4 * brain_turn + 0.35 * upwind_turn + 0.25 * seek_turn
            forward = max(0.45, conn_fwd)
            return turn, forward

        if behavior == "forage":
            turn = (1.0 - motivation) * wander_turn + motivation * brain_turn
            forward = (1.0 - motivation) * wander_fwd + motivation * max(conn_fwd, 0.35)
            return turn, forward

        ww = cfg.wander_weight
        bw = 1.0 - ww
        return ww * wander_turn + bw * brain_turn, ww * wander_fwd + bw * conn_fwd

    def _greedy_action(
        self, drives: dict, motivation: float, behavior: str
    ) -> tuple[float, float]:
        cfg = self.config
        self.state.motor_rates = np.zeros(len(self.motor_labels), dtype=np.float64)
        vis_l, vis_r = drives.get("visual", (0.0, 0.0))
        olf_l, olf_r = drives.get("olfactory", (0.0, 0.0))
        seek = 0.55 * (vis_l - vis_r) + 0.45 * (olf_l - olf_r)
        wander_turn, wander_fwd = self._wander()

        if behavior == "collapsed":
            return 0.0, 0.0
        if behavior == "stagger":
            wobble = self.rng.uniform(-1.0, 1.0)
            return cfg.stagger_turn_scale * wobble, cfg.stagger_speed_scale * 0.5
        if behavior == "desperate":
            turn = seek if abs(seek) > 0.05 else self.rng.uniform(-0.9, 0.9)
            return turn, 0.85
        if behavior == "escape":
            if self.state.loom_side == "left":
                return -cfg.escape_turn, cfg.escape_forward
            if self.state.loom_side == "right":
                return cfg.escape_turn, cfg.escape_forward
            return self.rng.choice([-1.0, 1.0]) * cfg.escape_turn, cfg.escape_forward
        if behavior == "pause":
            return 0.0, 0.0
        if behavior == "search":
            return cfg.search_turn_gain * math.sin(self.state.tick / 4.0), 0.2
        if behavior == "approach":
            return seek, 0.4
        if behavior == "upwind":
            desired = _wrap_angle(self.state.wind_angle + math.pi - self.state.heading)
            return max(-1.0, min(1.0, desired * 1.5)), 0.7
        if behavior == "forage":
            turn = (1.0 - motivation) * wander_turn + motivation * seek
            fwd = (1.0 - motivation) * wander_fwd + motivation * max(
                0.35, min(1.0, vis_l + vis_r + olf_l + olf_r)
            )
            return turn, fwd
        return wander_turn, wander_fwd

    def tick(self) -> WorldState:
        cfg = self.config
        st = self.state
        st.tick += 1

        # Don't drain while collapsed (already empty); soft-respawn restores energy.
        if st.starvation_stage != "collapsed":
            drain = cfg.energy_drain_per_tick
            if cfg.enable_flight_state and st.in_flight:
                drain *= cfg.flight_energy_mult
            st.energy = max(0.0, st.energy - drain)
        st.hunger = 1.0 - st.energy
        motivation = _smoothstep(cfg.seek_hunger_start, cfg.seek_hunger_full, st.hunger)
        st.motivation = motivation

        stage = self._update_starvation()

        self._update_wind()

        # Stochastic events (skip while collapsing)
        if stage != "collapsed":
            if st.loom_timer <= 0 and self.rng.random() < cfg.loom_prob_per_tick:
                self.trigger_loom()
            if (
                stage == "ok"
                and st.loom_timer <= 0
                and st.pause_timer <= 0
                and st.search_timer <= 0
                and not st.in_flight
                and st.hunger < cfg.pause_suppress_when_hunger
                and self.rng.random() < cfg.pause_prob_per_tick
            ):
                st.pause_timer = self.rng.randint(cfg.pause_min_ticks, cfg.pause_max_ticks)

        drives = self._sense_all(motivation)
        st.sense = {k: 0.5 * (v[0] + v[1]) for k, v in drives.items()}

        # Emergent pause: low prior forward drive + flat odor → groom/stop-and-go.
        if (
            stage == "ok"
            and st.loom_timer <= 0
            and st.pause_timer <= 0
            and st.search_timer <= 0
            and not st.in_flight
            and motivation < 0.55
            and float(st.forward_drive) < cfg.pause_low_drive
            and abs(float(st.odor) - float(self._prev_odor)) < cfg.pause_low_odor_grad
            and float(st.odor) < 0.25
            and self.rng.random() < cfg.pause_emergent_prob
        ):
            st.pause_timer = self.rng.randint(cfg.pause_min_ticks, cfg.pause_max_ticks)

        if st.loom_timer > 0:
            st.loom_timer -= 1
            if st.loom_timer <= 0:
                st.threat_x = st.threat_y = None
                st.threat_radius = 0.0
        if st.pause_timer > 0:
            st.pause_timer -= 1
        if st.search_timer > 0:
            st.search_timer -= 1

        behavior = self._choose_behavior(motivation, drives)
        st.behavior = behavior
        self._update_flight_state(behavior)

        if behavior == "collapsed":
            st.motor_rates = np.zeros(len(self.motor_labels), dtype=np.float64)
            turn, forward = 0.0, 0.0
        elif cfg.controller == Controller.CONNECTOME.value:
            turn, forward = self._connectome_action(drives, motivation, behavior)
        elif cfg.controller == Controller.GREEDY.value:
            turn, forward = self._greedy_action(drives, motivation, behavior)
        else:
            st.motor_rates = np.zeros(len(self.motor_labels), dtype=np.float64)
            turn, forward = self.rng.uniform(-1, 1), self.rng.uniform(0.2, 1.0)

        turn, forward = self._apply_flight_locomotion(turn, forward, behavior)

        st.turn_drive = turn
        st.forward_drive = forward
        if not cfg.wrap_world:
            # No dedicated wall sensors on the connectome — add a soft avoid bias
            # so walled sandboxes don't just pin the fly against edges.
            # Slightly weaker wall avoid in air (still bounce at bounds).
            wall_scale = 0.55 if st.in_flight else 1.0
            turn = max(-1.5, min(1.5, turn + wall_scale * self._wall_avoid_turn()))
            st.turn_drive = turn
        st.heading = _wrap_angle(st.heading + cfg.turn_rate * turn)
        speed = cfg.move_speed * max(0.0, forward)
        nx = st.fly_x + speed * math.cos(st.heading)
        ny = st.fly_y + speed * math.sin(st.heading)
        if cfg.wrap_world:
            st.fly_x = nx % cfg.grid_size
            st.fly_y = ny % cfg.grid_size
        else:
            lo = cfg.wall_margin
            hi = cfg.grid_size - cfg.wall_margin
            if nx < lo:
                nx = lo
                st.heading = _wrap_angle(math.pi - st.heading)
            elif nx > hi:
                nx = hi
                st.heading = _wrap_angle(math.pi - st.heading)
            if ny < lo:
                ny = lo
                st.heading = _wrap_angle(-st.heading)
            elif ny > hi:
                ny = hi
                st.heading = _wrap_angle(-st.heading)
            st.fly_x = min(hi, max(lo, nx))
            st.fly_y = min(hi, max(lo, ny))

        self._prev_odor = float(st.odor)
        self._prev_heading = float(st.heading)

        cx, cy = int(st.fly_x) % cfg.grid_size, int(st.fly_y) % cfg.grid_size
        remaining = []
        eaten_now = False
        for fx, fy in st.foods:
            can_eat = (
                st.hunger >= cfg.eat_min_hunger
                and behavior not in ("escape", "collapsed")
            )
            if fx == cx and fy == cy and not eaten_now and can_eat:
                st.foods_eaten.append(FoodEvent(tick=st.tick, x=fx, y=fy))
                st.energy = min(1.0, st.energy + cfg.energy_per_food)
                st.hunger = 1.0 - st.energy
                st.empty_ticks = 0
                st.starvation_stage = "ok"
                st.search_timer = cfg.search_duration
                if cfg.food_respawn:
                    st.pending_respawns.append(st.tick + cfg.food_respawn_delay)
                eaten_now = True
            else:
                remaining.append((fx, fy))
        st.foods = remaining

        still = []
        for due in st.pending_respawns:
            if st.tick >= due:
                self._spawn_food()
            else:
                still.append(due)
        st.pending_respawns = still
        return st
