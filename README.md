# prettyFlyfora — runnable sim package

FlyWire-driven pixel fly sim (2D grid + optional 3D follow-cam). This folder is a
**minimal runtime** — enough to run the sim, not the mapping / characterization tooling.

## Setup

```bash
cd package
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Needs a working OpenGL stack for `--view 3d` (and a display for windowed mode).

## Run

```bash
# 2D (default) — reduced connectome, smooth interactive
.venv/bin/python scripts/run_phase3.py

# 3D follow-cam (presentation layer; physics still 2D-grid)
.venv/bin/python scripts/run_phase3.py --view 3d

# Full brain (~139k neurons) — slower, uses sim atlas motor roles
.venv/bin/python scripts/run_phase3.py --network full --view 3d

# Headless smoke test
.venv/bin/python scripts/run_phase3.py --network reduced --headless --ticks 50
```

### Useful knobs

| Flag | Default | Meaning |
|------|---------|---------|
| `--network reduced\|full` | `reduced` | Connectome size (full is much slower) |
| `--view 2d\|3d` | `2d` | Renderer |
| `--fps` | `20` | Windowed frame rate |
| `--energy-hours` | `1.0` | Hours from full energy → empty at `--fps` (no food) |
| `--flight` / `--no-flight` | on | Stage D real flight (voluntary cruise + escape). Off = walk only |
| `--controller connectome\|greedy\|random` | `connectome` | Drive mode |
| `--seed` | `0` | RNG seed |

### Keys (windowed)

- `Esc` / `Q` — quit  
- `R` — reset  
- `C` — cycle controller  
- `Space` — pause  
- `L` — force loom / escape  
- **3D:** mouse drag orbit · scroll zoom  

## Flight (Stage D)

Default is **real walk/fly**, not walk-only with a cosmetic hop:

- **Wander / forage / upwind** can **take off** into sustained cruise when drive + energy are high and food isn’t right underfoot.
- **Escape** (loom / `L`) still does the surprise jump with a speed boost.
- **While flying:** faster track speed, softer turns, higher energy cost, cruise altitude for the 3D mesh.
- **Land** near food, on pause/approach, when tired, or after a max air time; short cooldown before the next voluntary takeoff.

Toggle with `--no-flight` if you want ground-only. Panel shows `loco: walk|flight` and altitude.

## Motor mapping: defaults vs remapping

The sim does **not** control the fly with one neuron → one muscle. Descending neurons
(DNs / MOTOR) are decoded into **turn** (left/right population vote) and **forward**
(role-weighted pools). How each DN type gets a role depends on which network you load.

### Default mapping (`--network reduced`)

The small network has **18 MOTOR** cells from a short hand-curated catalogue. Roles are
hard-coded in `flybrain/motor_decode.py`:

| Exact type | Role | Effect |
|------------|------|--------|
| `DNb05`, `DNg75`, `DNg100`, `DNg74_a` | thrust | push forward |
| `DNa02`, `DNa03`, `DNp103`, `DNge037` | steer | turn via L/R vote only |
| `DNpe042`, `DNpe053` | brake | reduce forward |
| `aSP22` | arousal | mild forward boost |

Anything else falls back to **prefix heuristics** (e.g. `DNb*` → thrust, `DNa*` → steer,
`DNpe*` → brake). Phase-2 stimulation weights (`motor_stimulation_summary.csv`) still
scale how strongly each cell counts (responsive / weak / dud).

On startup you should see: `SIM atlas: not loaded (prefix/hand roles only)`.

### Remapping for the full brain (`--network full`)

The full connectome has **~1,301 MOTOR** neurons across **~471 types** — too many to
hand-label. I remapped them offline, then baked the result into this package as
`data/processed/phase2/full/dn_sim_atlas.csv`. The live decoder loads that file
automatically.

How the remapping was done (in the research repo, not shipped here):

1. **Multi-modal sensory tagging** — rebuild the full graph so SENSORY includes visual,
   olfactory, wind, taste, and loom (LPLC2), not vision-only.
2. **Fingerprint stimulation** — drive each modality and record every MOTOR type’s
   response profile (plus L/R bias and top presynaptic partners).
3. **Discovery + fine atlas** — assign primary/secondary modalities, strength tiers,
   clusters, and partner pathway tags (JO, AL/PN, LPLC/LC4, etc.).
4. **Partner-validated shortlists** (high confidence only):
   - **escape** — loom stim + LPLC/LC partners  
   - **wind_jo** — wind stim + JO/AMMC partners  
   - **taste_contact** — taste stim  
   - **odor_al** — odor stim + AL/LH/PN partners  
   Putative piles (stim yes, partners weak) were kept separate at lower weight.
5. **Full sim atlas** — map *every* type into a decode role with a confidence tier and
   weight scale, so the sim isn’t limited to the shortlist:

| Tier | Meaning | Weight scale |
|------|---------|-------------:|
| `hand` | original exact `_DN_ROLE` types | 1.0 |
| `curated` | partner-validated shortlist | 1.0 |
| `putative` | stim-positive, weaker partners | 0.6 |
| `stim` | parked but fingerprint-driven | 0.3–0.5 |
| `prefix` | family heuristic fallback | 0.35 |
| `silent` | no reliable drive | 0.0 |

Decode-facing roles include ethology buckets (`escape`, `wind`, `odor`, `taste`, `walk`)
plus the legacy ones (`thrust`, `steer`, `brake`, `arousal`). Ethology roles contribute
to **forward** with their own gains; **steer** still only affects turn.

On startup you should see something like:
`SIM atlas: dn_sim_atlas.csv  mapped=1301/1301  roles: wind=…, escape=…, …`

**Caveat:** these are connectome + stim hypotheses wired for a playable sim, not
ground-truth ethology. Curated tiers are the ones we’d trust first; putative/stim are
included so the full MOTOR population can still move the fly.

## What’s included

- `flybrain/` — rate network, pixel world, motor decode, 3D view  
- `scripts/run_phase3.py` — launcher  
- `data/processed/` — `reduced_*` + `full_*` parquet connectomes, Phase-2 motor calibration, `dn_sim_atlas.csv` (full)

## What’s not included

Build / mapping scripts, characterization atlases, raw FlyWire dumps, run logs, tests.
(The remapping pipeline lived upstream; this package only ships the finished atlas CSV.)

## Size note

Most of the zip is `full_edges.parquet` (~38 MB). For a smaller share, delete the `full_*` files and only use `--network reduced`.

## Credits & data attribution

This sim is built on the **FlyWire Brain Dataset (FAFB v783)** — the female adult fruit fly
whole-brain connectome — obtained via **FlyWire Codex** ([codex.flywire.ai](https://codex.flywire.ai/)).

Please credit the FlyWire Consortium and cite the resource papers when sharing or
publishing work that uses this package:

1. **Wiring diagram** — Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature (2024).  
   https://doi.org/10.1038/s41586-024-07558-y  
2. **Annotations / cell types** — Schlegel et al., *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature (2024).  
   https://doi.org/10.1038/s41586-024-07686-5  
3. **Codex** (data explorer / download portal) — cite as:  
   http://dx.doi.org/10.13140/RG.2.2.35928.67844  

Dataset portal: [codex.flywire.ai](https://codex.flywire.ai/) · About / citation notes:  
[About FlyWire](https://codex.flywire.ai/about_flywire)

prettyFlyfora adds a rate-network ethogram sim and motor decoding on top of a
subgraph of that connectome; it is **not** an official FlyWire product. Synaptic
weights and neuron labels originate from FlyWire/FAFB v783; behaviour, decode roles,
and the 3D fly mesh are this project’s additions.
