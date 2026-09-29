#!/usr/bin/env python3
"""
Phase 3 — pixel grid fly sim driven by the connectome.

Usage:
  .venv/bin/python scripts/run_phase3.py
  .venv/bin/python scripts/run_phase3.py --view 3d
  .venv/bin/python scripts/run_phase3.py --network full --headless --ticks 50
  .venv/bin/python scripts/run_phase3.py --controller greedy
  .venv/bin/python scripts/run_phase3.py --headless --ticks 500

Keys (windowed 2d/3d):
  Esc / Q  quit
  R        reset
  C        cycle controller (connectome → greedy → random)
  Space    pause
  L        force a loom / escape event
  (3d)     mouse drag orbit · wheel zoom
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flybrain.pixel_sim import PixelSim, SimConfig  # noqa: E402


# Visual palette — warm amber field, teal fly, coral food (avoid purple-AI defaults).
COLOR_BG = (28, 24, 20)
COLOR_GRID = (40, 34, 28)
COLOR_FLY = (80, 200, 190)
COLOR_FLY_HUNGRY = (220, 140, 70)
COLOR_FLY_ESCAPE = (230, 70, 90)
COLOR_FLY_STAGGER = (160, 100, 80)
COLOR_FLY_DEAD = (70, 60, 55)
COLOR_FLY_HEAD = (220, 240, 230)
COLOR_FOOD = (220, 90, 60)
COLOR_THREAT = (180, 50, 40)
COLOR_PANEL = (18, 16, 14)
COLOR_TEXT = (230, 220, 200)
COLOR_BAR_L = (90, 180, 120)
COLOR_BAR_R = (220, 160, 70)
COLOR_ENERGY = (90, 180, 120)
COLOR_HUNGER = (220, 120, 70)
COLOR_MUTED = (140, 130, 120)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 pixel fly sim")
    p.add_argument(
        "--network",
        choices=["reduced", "full"],
        default="reduced",
        help="Load reduced_* or full_* connectome (full is much slower)",
    )
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--food", type=int, default=8)
    p.add_argument("--controller", choices=["connectome", "greedy", "random"], default="connectome")
    p.add_argument("--neural-steps", type=int, default=8)
    p.add_argument("--cell", type=int, default=10, help="pixel size of one grid cell")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sensory-gain",
        type=float,
        default=1.0,
        help="Multiplier for sensory modality drives (visual/olfactory/wind/taste/loom)",
    )
    p.add_argument(
        "--sensory-clamp",
        type=float,
        default=2.5,
        help="Clamp max applied when modality drives are written into sensory neurons",
    )
    p.add_argument(
        "--wander-weight",
        type=float,
        default=0.65,
        help="Blend weight for idle wander vs connectome on default wander behavior",
    )
    p.add_argument(
        "--seek-hunger-start",
        type=float,
        default=0.20,
        help="Hunger level where seek motivation begins rising (lower = earlier seeking)",
    )
    p.add_argument(
        "--energy-per-food",
        type=float,
        default=0.40,
        help="Energy restored per food (lower = hungrier cycle)",
    )
    p.add_argument(
        "--energy-hours",
        type=float,
        default=1.0,
        help="Wall-clock hours from full energy→empty at --fps (no food). Default 1.0",
    )
    p.add_argument(
        "--seek-assist",
        type=float,
        default=0.25,
        help="connectome_seek_assist blend toward vision/odor taxis (lower = more brain-led)",
    )
    p.add_argument(
        "--forage-motivation-threshold",
        type=float,
        default=0.15,
        help="Motivation required to enter forage behavior",
    )
    p.add_argument(
        "--relative-motor-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize motor rates by peak before decode (default: on)",
    )
    p.add_argument(
        "--view",
        choices=["2d", "3d"],
        default="2d",
        help="Windowed renderer: classic pixel grid (2d) or follow-cam 3D presentation",
    )
    p.add_argument(
        "--flight",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stage D real flight (voluntary cruise + escape takeoff). --no-flight = walk only",
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument("--ticks", type=int, default=0, help="if >0, run this many ticks then exit")
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="CSV log path (default: data/processed/phase3/{network}/run_log.csv)",
    )
    return p.parse_args()


def ensure_font(pygame, size: int):
    return pygame.font.SysFont("DejaVu Sans", size)


def _top_motor_indices(rates: np.ndarray, k: int) -> list[int]:
    if rates.size == 0:
        return []
    k = min(k, rates.size)
    return np.argsort(-rates)[:k].tolist()


def draw(
    pygame,
    screen,
    font,
    small,
    sim: PixelSim,
    cell: int,
    panel_w: int,
    paused: bool,
) -> None:
    cfg = sim.config
    st = sim.state
    g = cfg.grid_size
    screen.fill(COLOR_BG)

    # Grid world
    for x in range(g):
        for y in range(g):
            rect = pygame.Rect(x * cell, y * cell, cell, cell)
            pygame.draw.rect(screen, COLOR_GRID, rect, 1)

    for fx, fy in st.foods:
        pygame.draw.rect(
            screen,
            COLOR_FOOD,
            pygame.Rect(fx * cell, fy * cell, cell, cell),
        )

    # Looming threat disc
    if st.threat_x is not None and st.threat_y is not None and st.threat_radius > 0:
        tr = max(cell, int(st.threat_radius * cell * 0.35))
        pygame.draw.circle(
            screen,
            COLOR_THREAT,
            (int(st.threat_x * cell), int(st.threat_y * cell)),
            tr,
            width=max(1, cell // 3),
        )

    cx, cy = int(st.fly_x) % g, int(st.fly_y) % g
    if st.behavior == "escape":
        fly_col = COLOR_FLY_ESCAPE
    elif st.behavior == "collapsed":
        fly_col = COLOR_FLY_DEAD
    elif st.behavior in ("stagger", "desperate"):
        fly_col = COLOR_FLY_STAGGER if st.behavior == "stagger" else COLOR_FLY_HUNGRY
    else:
        h = float(st.hunger)
        fly_col = tuple(
            int(COLOR_FLY[i] * (1 - h) + COLOR_FLY_HUNGRY[i] * h) for i in range(3)
        )
    pygame.draw.rect(
        screen,
        fly_col,
        pygame.Rect(cx * cell, cy * cell, cell, cell),
    )
    import math

    hx = int(st.fly_x + 0.9 * math.cos(st.heading)) % g
    hy = int(st.fly_y + 0.9 * math.sin(st.heading)) % g
    pygame.draw.rect(
        screen,
        COLOR_FLY_HEAD,
        pygame.Rect(hx * cell, hy * cell, cell, cell),
    )

    panel = pygame.Rect(g * cell, 0, panel_w, g * cell)
    pygame.draw.rect(screen, COLOR_PANEL, panel)

    y = 12
    cal = sim.motor_cal
    dec = st.motor_decode
    lines = [
        f"network: {cfg.network}",
        f"controller: {cfg.controller}",
        f"behavior: {st.behavior}",
        f"loco: {getattr(st, 'locomotion', 'walk')}  alt: {getattr(st, 'altitude', 0):.2f}",
        f"starve: {st.starvation_stage}  deaths: {st.deaths}",
        f"tick: {st.tick}  eaten: {len(st.foods_eaten)}",
        f"energy: {st.energy:.2f}  hunger: {st.hunger:.2f}",
        f"odor: {st.odor:.2f}  wind: {st.wind_angle:+.2f}",
        f"turn/fwd: {st.turn_drive:+.2f} / {st.forward_drive:.2f}",
        "paused" if paused else "running",
    ]
    for line in lines:
        screen.blit(font.render(line, True, COLOR_TEXT), (g * cell + 12, y))
        y += 18

    bar_x = g * cell + 12
    bar_w = panel_w - 24
    screen.blit(small.render("energy", True, COLOR_MUTED), (bar_x, y))
    y += 12
    pygame.draw.rect(screen, (50, 44, 38), pygame.Rect(bar_x, y, bar_w, 8), border_radius=2)
    pygame.draw.rect(
        screen, COLOR_ENERGY, pygame.Rect(bar_x, y, max(1, int(bar_w * st.energy)), 8), border_radius=2
    )
    y += 14
    screen.blit(small.render("hunger→seek", True, COLOR_MUTED), (bar_x, y))
    y += 12
    pygame.draw.rect(screen, (50, 44, 38), pygame.Rect(bar_x, y, bar_w, 8), border_radius=2)
    pygame.draw.rect(
        screen, COLOR_HUNGER, pygame.Rect(bar_x, y, max(1, int(bar_w * st.motivation)), 8), border_radius=2
    )
    y += 14

    screen.blit(small.render("SENSORY modalities", True, COLOR_MUTED), (bar_x, y))
    y += 14
    for mod in ("visual", "olfactory", "mechano_wind", "gustatory", "visual_loom"):
        val = float(st.sense.get(mod, 0.0))
        screen.blit(small.render(mod, True, COLOR_TEXT), (bar_x, y))
        y += 11
        pygame.draw.rect(screen, (50, 44, 38), pygame.Rect(bar_x, y, bar_w, 6), border_radius=2)
        pygame.draw.rect(
            screen,
            COLOR_BAR_L,
            pygame.Rect(bar_x, y, max(1, int(bar_w * min(1.0, val))), 6),
            border_radius=2,
        )
        y += 9

    y += 4
    n_motor = len(sim.motor_labels)
    screen.blit(
        small.render(f"MOTOR decode ({n_motor} neurons)", True, COLOR_MUTED),
        (g * cell + 12, y),
    )
    y += 14

    if dec is not None:
        agg_lines = [
            f"active: {dec.n_active}  w_sum={dec.active_weight:.3f}",
            f"L/R mass: {dec.turn_left_mass:.3f} / {dec.turn_right_mass:.3f}",
            f"thrust/brake/arousal: {dec.thrust:.2f} / {dec.brake:.2f} / {dec.arousal:.2f}",
            f"cal: resp={cal.n_responsive} weak={cal.n_weak} dud={cal.n_dud}",
        ]
        for line in agg_lines:
            screen.blit(small.render(line, True, COLOR_TEXT), (bar_x, y))
            y += 12

    show_bars = min(n_motor, cfg.motor_panel_max)
    if show_bars and st.motor_rates.size:
        if n_motor > cfg.motor_panel_max:
            screen.blit(
                small.render(f"top {show_bars} by rate", True, COLOR_MUTED),
                (bar_x, y),
            )
            y += 12
            indices = _top_motor_indices(st.motor_rates, show_bars)
        else:
            indices = list(range(n_motor))

        max_r = max(float(st.motor_rates.max()), 1e-6)
        for i in indices:
            label = sim.motor_labels[i]
            rate = float(st.motor_rates[i])
            side = sim.motor_sides[i]
            cat = sim.motor_cal.categories[i]
            screen.blit(
                small.render(f"{label} [{cat[:3]}]", True, COLOR_TEXT),
                (bar_x, y),
            )
            y += 11
            pygame.draw.rect(screen, (50, 44, 38), pygame.Rect(bar_x, y, bar_w, 5), border_radius=2)
            frac = float(np.clip(rate / max_r, 0, 1))
            col = COLOR_BAR_L if side == "left" else COLOR_BAR_R
            pygame.draw.rect(
                screen, col, pygame.Rect(bar_x, y, max(1, int(bar_w * frac)), 5), border_radius=2
            )
            y += 8

    y += 6
    for chunk in _wrap(cfg.note_left_skew, 34):
        screen.blit(small.render(chunk, True, COLOR_MUTED), (g * cell + 12, y))
        y += 12


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def append_log(path: Path, sim: PixelSim, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    st = sim.state
    dec = st.motor_decode
    row = {
        "tick": st.tick,
        "network": sim.config.network,
        "controller": sim.config.controller,
        "behavior": st.behavior,
        "locomotion": getattr(st, "locomotion", "walk"),
        "in_flight": int(bool(getattr(st, "in_flight", False))),
        "altitude": f"{getattr(st, 'altitude', 0.0):.4f}",
        "starvation_stage": st.starvation_stage,
        "deaths": st.deaths,
        "empty_ticks": st.empty_ticks,
        "fly_x": f"{st.fly_x:.3f}",
        "fly_y": f"{st.fly_y:.3f}",
        "heading": f"{st.heading:.4f}",
        "energy": f"{st.energy:.4f}",
        "hunger": f"{st.hunger:.4f}",
        "motivation": f"{st.motivation:.4f}",
        "odor": f"{st.odor:.4f}",
        "wind_angle": f"{st.wind_angle:.4f}",
        "turn_drive": f"{st.turn_drive:.4f}",
        "forward_drive": f"{st.forward_drive:.4f}",
        "n_food": len(st.foods),
        "foods_eaten_total": len(st.foods_eaten),
        "motor_n_active": dec.n_active if dec else 0,
        "motor_turn": f"{dec.turn:.6f}" if dec else "0",
        "motor_forward": f"{dec.forward:.6f}" if dec else "0",
        "motor_left_mass": f"{dec.turn_left_mass:.6f}" if dec else "0",
        "motor_right_mass": f"{dec.turn_right_mass:.6f}" if dec else "0",
        "sensory_gain_used": f"{st.sensory_gain_used:.4f}",
        "sensory_clamp_used": f"{st.sensory_clamp_used:.4f}",
        "sensory_pre_clamp_max": f"{st.sensory_pre_clamp_max:.6f}",
        "sensory_post_clamp_max": f"{st.sensory_post_clamp_max:.6f}",
        "conn_turn": f"{st.conn_turn:.6f}",
        "conn_forward": f"{st.conn_forward:.6f}",
        "seek_turn": f"{st.seek_turn:.6f}",
        "brain_turn": f"{st.brain_turn:.6f}",
        "wander_turn": f"{st.wander_turn:.6f}",
        "wander_forward": f"{st.wander_forward:.6f}",
        "connectome_seek_assist_used": f"{st.connectome_seek_assist_used:.6f}",
        "satiation": f"{st.energy:.4f}",
    }
    for mod, val in st.sense.items():
        row[f"sense_{mod}"] = f"{float(val):.4f}"

    # Per-neuron columns only for reduced network (full → aggregate only).
    if sim.config.network == "reduced":
        for lab, rate in zip(sim.motor_labels, st.motor_rates):
            row[f"motor_{lab}"] = f"{float(rate):.6f}"

    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_headless(sim: PixelSim, ticks: int, log_path: Path) -> None:
    if log_path.exists():
        log_path.unlink()
    import time

    t0 = time.perf_counter()
    for t in range(ticks):
        sim.tick()
        append_log(log_path, sim, write_header=(t == 0))
    elapsed = time.perf_counter() - t0
    print(
        f"headless done: network={sim.config.network} ticks={ticks} "
        f"eaten={len(sim.state.foods_eaten)} controller={sim.config.controller} "
        f"wall={elapsed:.2f}s ({elapsed / max(ticks, 1) * 1000:.1f} ms/tick) "
        f"log={log_path}"
    )


def run_windowed(sim: PixelSim, args: argparse.Namespace) -> None:
    import pygame

    pygame.init()
    pygame.display.set_caption(f"prettyFlyfora — Phase 3 ({sim.config.network})")
    cell = args.cell
    g = sim.config.grid_size
    panel_w = 300
    screen = pygame.display.set_mode((g * cell + panel_w, g * cell))
    clock = pygame.time.Clock()
    font = ensure_font(pygame, 16)
    small = ensure_font(pygame, 12)

    if args.log.exists():
        args.log.unlink()
    paused = False
    controllers = ["connectome", "greedy", "random"]
    first_log = True
    running = True
    max_ticks = args.ticks if args.ticks > 0 else None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim.reset()
                    first_log = True
                    if args.log.exists():
                        args.log.unlink()
                elif event.key == pygame.K_l:
                    sim.trigger_loom()
                elif event.key == pygame.K_c:
                    i = controllers.index(sim.config.controller)
                    sim.config.controller = controllers[(i + 1) % len(controllers)]
                    sim.reset()
                    first_log = True
                    if args.log.exists():
                        args.log.unlink()
                        first_log = True

        if not paused:
            sim.tick()
            append_log(args.log, sim, write_header=first_log)
            first_log = False
            if max_ticks is not None and sim.state.tick >= max_ticks:
                running = False

        draw(pygame, screen, font, small, sim, cell, panel_w, paused)
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    print(
        f"quit: network={sim.config.network} ticks={sim.state.tick} "
        f"eaten={len(sim.state.foods_eaten)} log={args.log}"
    )


def main() -> None:
    args = parse_args()
    log_path = args.log or (
        ROOT / "data" / "processed" / "phase3" / args.network / "run_log.csv"
    )
    fps = max(1, int(args.fps))
    energy_hours = max(1e-6, float(args.energy_hours))
    energy_drain = 1.0 / (fps * 3600.0 * energy_hours)
    cfg = SimConfig(
        grid_size=args.grid,
        n_food=args.food,
        neural_steps_per_tick=args.neural_steps,
        controller=args.controller,
        seed=args.seed,
        network=args.network,
        processed_dir=ROOT / "data" / "processed",
        sensory_gain_mult=args.sensory_gain,
        sensory_clamp_max=args.sensory_clamp,
        wander_weight=args.wander_weight,
        seek_hunger_start=args.seek_hunger_start,
        energy_per_food=args.energy_per_food,
        energy_drain_per_tick=energy_drain,
        connectome_seek_assist=args.seek_assist,
        forage_motivation_threshold=args.forage_motivation_threshold,
        relative_motor_decode=args.relative_motor_decode,
        enable_flight_state=bool(args.flight),
        # 3D sandbox has visible walls — bounce instead of toroidal wrap.
        wrap_world=(args.view != "3d"),
    )
    print(
        f"Energy: ~{energy_hours:.2f}h from full→empty @ {fps} FPS "
        f"(drain={energy_drain:.2e}/tick); food still restores energy"
    )
    print(
        f"Flight: {'ON (Stage D voluntary cruise + escape)' if args.flight else 'OFF (walk only)'}"
    )
    print(cfg.note_left_skew)
    print(
        "Senses: visual | olfactory(odor plume) | mechano_wind | gustatory | visual_loom(LPLC2)"
    )
    print(
        "Behaviours: wander | forage | upwind | approach | pause | escape | search"
    )
    if args.network == "full":
        print(
            "NOTE: full-brain sim is slow (~seconds/tick). "
            "Use --headless --ticks N for benchmarking."
        )

    print(
        f"CLI → SimConfig: network={cfg.network} sensory_gain_mult={cfg.sensory_gain_mult} "
        f"sensory_clamp_max={cfg.sensory_clamp_max}"
    )
    sim = PixelSim(cfg)
    assert sim.config.network == args.network
    assert sim.config.sensory_gain_mult == args.sensory_gain
    assert sim.config.sensory_clamp_max == args.sensory_clamp
    mods = (
        sim.net.neurons.loc[sim.net.neurons.role == "SENSORY", "modality"]
        .value_counts()
        .to_dict()
    )
    print(f"network={args.network}  N={len(sim.net.root_ids):,}")
    print(f"SENSORY modalities: {mods}")
    print(
        f"MOTOR: {len(sim.motor_labels)} neurons "
        f"(cal resp={sim.motor_cal.n_responsive} weak={sim.motor_cal.n_weak} "
        f"dud={sim.motor_cal.n_dud})"
    )
    if sim.motor_cal.summary_path:
        print(f"MOTOR calibration: {sim.motor_cal.summary_path}")
    else:
        print("MOTOR calibration: defaults (no Phase-2 summary found)")

    if sim.motor_cal.sim_atlas_path is not None:
        from collections import Counter

        role_counts = Counter(sim.motor_roles.astype(str))
        top = ", ".join(f"{r}={n}" for r, n in role_counts.most_common(8))
        print(
            f"SIM atlas: {sim.motor_cal.sim_atlas_path.name}  "
            f"mapped={sim.motor_cal.n_sim_mapped}/{len(sim.motor_labels)}  "
            f"roles: {top}"
        )
    else:
        print("SIM atlas: not loaded (prefix/hand roles only — build dn_sim_atlas.csv for full)")

    if args.network == "reduced":
        print(f"MOTOR labels: {', '.join(sim.motor_labels)}")
        print(
            "DN roles: "
            + ", ".join(f"{lab}={role}" for lab, role in zip(sim.motor_labels, sim.motor_roles))
        )
    else:
        print("MOTOR decode: population weighted L/R vote + role pools (see flybrain/motor_decode.py)")

    if args.headless:
        ticks = args.ticks if args.ticks > 0 else 500
        run_headless(sim, ticks, log_path)
    elif args.view == "3d":
        from flybrain.view3d import run_3d  # noqa: E402

        print("Launching 3D follow-cam view (presentation layer; sim stays 2D-grid).")
        print("Prefer --network reduced for interactive frame rate.")
        run_3d(
            sim,
            fps=args.fps,
            ticks=args.ticks,
            log_path=log_path,
            append_log=append_log,
        )
    else:
        run_windowed(sim, argparse.Namespace(**{**vars(args), "log": log_path}))


if __name__ == "__main__":
    main()
