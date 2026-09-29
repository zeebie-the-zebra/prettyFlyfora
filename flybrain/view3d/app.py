"""3D follow-cam window loop for Phase 3."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from flybrain.pixel_sim import PixelSim

from .anim import AnimState, update_anim
from .camera import FollowCamera, look_at, perspective
from .fly_model import load_fly_model
from .pose import from_state
from . import world as world_geo

VERT = """
#version 330
uniform mat4 mvp;
in vec3 in_pos;
in vec3 in_color;
out vec3 v_color;
void main() {
    v_color = in_color;
    gl_Position = mvp * vec4(in_pos, 1.0);
}
"""

FRAG = """
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    // Simple height-ish shading via depth approximation in color
    f_color = vec4(v_color, 1.0);
}
"""


def run_3d(
    sim: PixelSim,
    *,
    fps: int = 20,
    ticks: int = 0,
    log_path: Path | None = None,
    append_log=None,
    width: int = 1100,
    height: int = 720,
) -> None:
    """Open a pygame+ModernGL window; tick sim and render follow-cam view."""
    try:
        import pygame
        import moderngl
    except ImportError as exc:
        raise SystemExit(
            "3D view requires pygame and moderngl. Install with:\n"
            "  .venv/bin/pip install moderngl\n"
            f"Original error: {exc}"
        ) from exc

    pygame.init()
    pygame.display.set_caption(f"prettyFlyfora — 3D ({sim.config.network})")
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("DejaVu Sans", 16)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    ctx.enable(moderngl.BLEND)
    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

    fly_model = load_fly_model()
    cam = FollowCamera()
    static = world_geo.build_static_world(sim.config.grid_size)

    def make_vao_mesh(pos: np.ndarray, col: np.ndarray, idx: np.ndarray):
        if len(pos) == 0:
            return None
        vbo = ctx.buffer(np.hstack([pos, col]).astype("f4").tobytes())
        ibo = ctx.buffer(idx.astype("i4").tobytes())
        return ctx.vertex_array(
            prog,
            [(vbo, "3f 3f", "in_pos", "in_color")],
            ibo,
        )

    def make_vao_lines(pos: np.ndarray, col: np.ndarray):
        if len(pos) == 0:
            return None
        vbo = ctx.buffer(np.hstack([pos, col]).astype("f4").tobytes())
        return ctx.vertex_array(prog, [(vbo, "3f 3f", "in_pos", "in_color")])

    ground_vao = make_vao_mesh(static["ground_pos"], static["ground_col"], static["ground_idx"])
    grid_vao = make_vao_lines(static["grid_pos"], static["grid_col"])
    wall_vao = make_vao_mesh(static["wall_pos"], static["wall_col"], static["wall_idx"])

    paused = False
    first_log = True
    running = True
    max_ticks = ticks if ticks > 0 else None
    controllers = ["connectome", "greedy", "random"]
    anim = AnimState()
    altitude = 0.12
    t0 = time.perf_counter()
    dragging = False
    last_mouse = (0, 0)

    # HUD texture (updated each frame)
    hud_tex = ctx.texture((width, 120), 4)
    hud_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    hud_prog = ctx.program(
        vertex_shader="""
            #version 330
            in vec2 in_pos;
            in vec2 in_uv;
            out vec2 v_uv;
            void main() {
                v_uv = in_uv;
                gl_Position = vec4(in_pos, 0.0, 1.0);
            }
        """,
        fragment_shader="""
            #version 330
            uniform sampler2D tex;
            in vec2 v_uv;
            out vec4 f_color;
            void main() {
                f_color = texture(tex, v_uv);
            }
        """,
    )
    # NDC quad top strip. UV v=1 at screen top so pygame (top-origin) text
    # matches after tobytes(..., flipped=True) → GL bottom-origin texture.
    hud_h = 120.0
    ndc_h = 2.0 * hud_h / height
    hud_verts = np.array(
        [
            -1.0, 1.0, 0.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0 - ndc_h, 1.0, 0.0,
            -1.0, 1.0, 0.0, 1.0,
            1.0, 1.0 - ndc_h, 1.0, 0.0,
            -1.0, 1.0 - ndc_h, 0.0, 0.0,
        ],
        dtype=np.float32,
    )
    hud_vbo = ctx.buffer(hud_verts.tobytes())
    hud_vao = ctx.vertex_array(hud_prog, [(hud_vbo, "2f 2f", "in_pos", "in_uv")])

    while running:
        dt = clock.tick(fps) / 1000.0
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
                    cam.reset_orbit()
                    altitude = 0.12
                    first_log = True
                    if log_path and log_path.exists():
                        log_path.unlink()
                elif event.key == pygame.K_l:
                    sim.trigger_loom()
                elif event.key == pygame.K_c:
                    i = controllers.index(sim.config.controller)
                    sim.config.controller = controllers[(i + 1) % len(controllers)]
                    sim.reset()
                    first_log = True
                    if log_path and log_path.exists():
                        log_path.unlink()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    last_mouse = event.pos
                elif event.button == 4:
                    cam.zoom(-0.08)
                elif event.button == 5:
                    cam.zoom(0.08)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                dx = event.pos[0] - last_mouse[0]
                dy = event.pos[1] - last_mouse[1]
                last_mouse = event.pos
                cam.orbit(-dx * 0.005, dy * 0.004)

        if not paused:
            sim.tick()
            if append_log and log_path is not None:
                append_log(log_path, sim, write_header=first_log)
                first_log = False
            if max_ticks is not None and sim.state.tick >= max_ticks:
                running = False

        st = sim.state
        pose = from_state(st, prev_altitude=altitude, dt=dt)
        altitude = pose.y
        now = time.perf_counter() - t0
        anim = update_anim(pose, time_s=now, prev=anim)

        eye, target = cam.update(pose, dt)
        aspect = width / max(height, 1)
        proj = perspective(55.0, aspect, 0.1, 200.0)
        view = look_at(eye, target)
        mvp = (proj @ view).astype("f4")
        prog["mvp"].write(mvp.T.tobytes())

        ctx.clear(0.11, 0.09, 0.08, 1.0)
        if ground_vao is not None:
            ground_vao.render()
        if wall_vao is not None:
            wall_vao.render()
        if grid_vao is not None:
            grid_vao.render(moderngl.LINES)

        food_pos, food_col, food_idx = world_geo.food_markers(st)
        food_vao = make_vao_mesh(food_pos, food_col, food_idx)
        if food_vao is not None:
            food_vao.render()
            food_vao.release()

        tpos, tcol = world_geo.threat_disc(st)
        threat_vao = make_vao_lines(tpos, tcol)
        if threat_vao is not None:
            threat_vao.render(moderngl.LINES)
            threat_vao.release()

        fpos, fcol, fidx = fly_model.mesh(pose, anim)
        fly_vao = make_vao_mesh(fpos, fcol, fidx)
        if fly_vao is not None:
            fly_vao.render()
            fly_vao.release()

        # HUD overlay
        hud_surf = pygame.Surface((width, 120), pygame.SRCALPHA)
        hud_surf.fill((18, 16, 14, 200))
        lines = [
            f"network={sim.config.network}  view=3d  {'PAUSED' if paused else 'running'}",
            f"behavior={st.behavior}  loco={getattr(st, 'locomotion', 'walk')}  "
            f"alt={pose.y:.2f}  starve={st.starvation_stage}  deaths={st.deaths}",
            f"tick={st.tick}  eaten={len(st.foods_eaten)}  energy={st.energy:.2f}  hunger={st.hunger:.2f}",
            f"turn/fwd={st.turn_drive:+.2f}/{st.forward_drive:.2f}  "
            f"flight_ticks={getattr(st, 'flight_ticks', 0)}  controller={sim.config.controller}",
            "Keys: drag orbit · wheel zoom · Space pause · R reset · L loom · C controller · Esc quit",
        ]
        for i, line in enumerate(lines):
            hud_surf.blit(font.render(line, True, (230, 220, 200)), (12, 8 + i * 20))
        raw = pygame.image.tobytes(hud_surf, "RGBA", True)
        hud_tex.write(raw)
        ctx.disable(moderngl.DEPTH_TEST)
        hud_tex.use(0)
        hud_prog["tex"] = 0
        hud_vao.render()
        ctx.enable(moderngl.DEPTH_TEST)

        pygame.display.flip()

    pygame.quit()
    print(
        f"quit 3d: network={sim.config.network} ticks={sim.state.tick} "
        f"eaten={len(sim.state.foods_eaten)}"
        + (f" log={log_path}" if log_path else "")
    )
