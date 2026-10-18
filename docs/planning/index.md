# Motion Planning

<div class="grid cards" markdown>

-   __OMPL frontend__

    ---

    All of [OMPL](https://ompl.kavrakilab.org/)'s mature sampling-based
    planners (30 exposed), driven by one uniform Python call. No glue
    code to swap planners — change a string.

-   __VAMP backend__

    ---

    Collision checking runs through [VAMP](https://github.com/KavrakiLab/vamp)'s
    SIMD-vectorised sphere pipeline. Point-cloud obstacles are
    broadphase-indexed.

-   __C++ hot path__

    ---

    Python calls `plan(start, goal)` once. Everything inside —
    sampling, NN queries, collision checks, path simplification —
    runs in C++ behind a single nanobind call.

</div>

## Architecture

```
         Python
           │  planner.plan(start, goal)
           ▼
    ┌─────────────────────────────────────────┐
    │     OMPL SimpleSetup  (C++)              │
    │  ┌────────────┐  ┌──────────────────┐    │
    │  │  Planner   │→ │ MotionValidator  │    │
    │  │  RRT-C,    │  │  checkMotion     │    │
    │  │  BIT*, …   │  └────────┬─────────┘    │
    │  └────────────┘           │              │
    │                           ▼              │
    │                  ┌─────────────────┐     │
    │                  │ VAMP SIMD       │     │
    │                  │  sphere-cloud   │     │
    │                  │  collision      │     │
    │                  └─────────────────┘     │
    └─────────────────────────────────────────┘
```

The frontend is vanilla OMPL — `ProjectedStateSpace`, `StateCostIntegralObjective`,
`SimpleSetup`. The collision checker is the only replaced component;
it swaps OMPL's per-state user callback for a VAMP pipeline that
batches many sphere/point checks into a single AVX2 kernel.

## Supported planners

The `planner_name` field of
[`PlannerConfig`](../api/types.md#wheelchair_planning.types.planning.PlannerConfig)
accepts any of:

=== "Feasibility (single-query, returns on first solution)"

    | Family | Planners |
    |---|---|
    | RRT | `rrtc`, `rrt`, `trrt`, `bitrrt` |
    | KPIECE | `kpiece`, `bkpiece`, `lbkpiece` |
    | PRM | `prm`, `lazyprm` |
    | Exploration | `est`, `biest`, `sbl`, `stride`, `pdst` |

=== "Asymptotically optimal (keep refining until budget)"

    | Family | Planners |
    |---|---|
    | RRT\* | `rrtstar`, `informed_rrtstar`, `rrtsharp`, `rrtxstatic` |
    | Informed trees | `bitstar`, `abitstar`, `aitstar`, `eitstar`, `blitstar` |
    | FMT | `fmt`, `bfmt` |
    | PRM\* | `prmstar`, `lazyprmstar` |
    | Near-optimal | `lbtrrt`, `spars`, `spars2` |

## Benchmarks

Median / p95 planning time of a single **search** (simplification and
interpolation disabled), N = 20 runs per cell, from the home pose to a
sampled collision-free goal around the table point cloud of
`examples/planning/motion.py`. Every run in every cell found a path.

### Feasibility planners — time to first solution

| Planner | Arm (7 DOF) | Base (3 DOF) | Whole body (10 DOF) |
|---|---:|---:|---:|
| `rrtc` | 2.25 / 4.38 ms | 1.14 / 1.23 ms | 2.19 / 3.26 ms |
| `rrt` | 6.48 / 1008.44 ms | 3.26 / 7.68 ms | 8.04 / 696.35 ms |
| `kpiece` | 6.60 / 34.11 ms | 2.22 / 7.43 ms | 7.54 / 31.83 ms |
| `bkpiece` | 8.69 / 15.99 ms | 3.89 / 7.48 ms | 49.24 / 98.79 ms |
| `lbkpiece` | 9.60 / 16.12 ms | 3.35 / 5.51 ms | 56.96 / 86.09 ms |
| `prm` | 18.57 / 46.76 ms | 109.83 / 257.67 ms | 11.71 / 405.92 ms |
| `lazyprm` | 4.35 / 26.90 ms | 1.19 / 2.33 ms | 4.29 / 14.95 ms |
| `est` | 6.03 / 46.79 ms | 4.33 / 14.86 ms | 11.80 / 37.19 ms |
| `biest` | 4.40 / 7.56 ms | 1.20 / 3.31 ms | 4.38 / 9.60 ms |
| `sbl` | 6.51 / 13.01 ms | 2.19 / 2.32 ms | 14.35 / 37.29 ms |
| `stride` | 6.58 / 49.77 ms | 2.19 / 5.41 ms | 14.89 / 32.18 ms |

### Asymptotically-optimal planners — median path cost after a 1.0 s budget

| Planner | Arm | Base | Whole body |
|---|---:|---:|---:|
| `rrtstar` | 11.33 | 7.08 | 17.53 |
| `bitstar` | 9.92 | 8.60 | 16.23 |
| `aitstar` | 10.84 | 9.61 | 21.26 |
| `eitstar` | 9.43 | 6.72 | 16.88 |
| `fmt` | 10.33 | 8.69 | 14.83 |

!!! note "Measurement"
    `pixi run -e dev python scripts/benchmarks/motion_planning.py`, commit
    `ed1939a` plus the benchmark's subgroup and scene fix, 2026-10-19,
    AMD EPYC 7543 (128 logical CPUs, shared with other jobs, load 5–11) ·
    Linux 6.8 · Python 3.14.7. p95 is the script's nearest-rank value.
    Raw output: `/nas/pollux/tianrun/wheelchair_flask/geo_benchmark.txt`.

## Minimal example

<video controls loop muted playsinline width="100%">
  <source src="../../assets/motion_planning.mp4" type="video/mp4">
</video>

```python
import numpy as np

from wheelchair_planning.wheelchair import HOME_JOINTS
from wheelchair_planning.planning import create_planner
from wheelchair_planning.types import PlannerConfig

planner = create_planner(
    "wheelchair_arm",
    config=PlannerConfig(planner_name="rrtc", time_limit=1.0),
    base_config=HOME_JOINTS.copy(),
    pointcloud=obstacle_cloud,           # (N, 3) np.float32
)

start = planner.extract_config(HOME_JOINTS)
goal = planner.sample_valid()
result = planner.plan(start, goal)
print(result.success, result.planning_time_ns * 1e-6, "ms")
```

## Configuring a plan

Every plan is driven by a single
[`PlannerConfig`](../api/types.md#wheelchair_planning.types.planning.PlannerConfig)
dataclass. The defaults are tuned for interactive use — a
millisecond-scale single-query plan with a dense, simplified path
— so you rarely need to touch anything except `planner_name` and
`time_limit`. The knobs that matter in practice:

```python
PlannerConfig(
    # Which OMPL planner to run — see "Supported planners" above.
    planner_name="rrtc",

    # Wall-clock ceiling for the search.  Feasibility planners return
    # on the first solution, so this is just a safety cap.  AO planners
    # (rrtstar, bitstar, …) run until the budget to refine cost.
    time_limit=1.0,

    # Inflation applied to every obstacle point in the VAMP broadphase.
    # Larger = more conservative, but slows the search.  0.01 m is a
    # good default for ~1 cm-resolution scans.
    point_radius=0.01,

    # After the raw path, run OMPL's SimpleSetup shortcutter.  Removes
    # jagged detours the sampler produced without changing homotopy.
    # Turn this OFF for constrained / cost planners — the default
    # shortcutter takes straight-line shortcuts that ignore custom
    # constraints and costs.
    simplify=True,

    # Resample the (simplified) path densely: each edge of length d
    # gets ceil(d * resolution) segments; resolution=0.0 falls back to
    # OMPL's default longest-valid-segment fraction, usually too sparse
    # for control.  The default (64.0) gives ~64 waypoints per unit of
    # state-space distance, which is smooth enough for a 100 Hz control
    # loop and robust to low-frequency replanning.
    interpolate=True,
    resolution=64.0,

    # Joint-range overrides by joint name, (lower, upper) in metres for
    # the virtual base joints and radians otherwise.  Joints left out
    # keep the range compiled into the robot model (the base spans
    # ±10 m); the planner samples only inside these ranges.
    joint_limits={"Joint_Virtual_X": (-2.0, 3.0), "Joint_Virtual_Y": (-1.5, 1.5)},
)
```

`planner.set_joint_limits(limits)` changes the ranges of an existing
planner with the same semantics: joints left out take the compiled
range (`None` restores every one), an unknown name or a range that is
not finite with `lower < upper` raises `ValueError`, and the ranges are
kept across `set_subgroup`. `planner.bounds` gives the active joints'
ranges as `(lower, upper)` arrays and `planner.joint_limits` every
joint's range by name.

Typical recipes:

=== "Interactive — fast feasibility"

    ```python
    PlannerConfig(
        planner_name="rrtc",          # returns on first solution
        time_limit=0.5,
        simplify=True,
        resolution=64.0,
    )
    ```

=== "High-quality, asymptotically optimal"

    ```python
    PlannerConfig(
        planner_name="bitstar",       # keeps refining until budget
        time_limit=2.0,
        simplify=True,
        resolution=128.0,             # denser output for smoother control
    )
    ```

=== "Constrained / cost planner"

    ```python
    PlannerConfig(
        planner_name="rrtstar",
        time_limit=5.0,
        simplify=False,               # the shortcutter ignores custom cost/constraint
        resolution=64.0,
    )
    ```

## Post-hoc simplify / interpolate

`simplify` and `interpolate` run inside `plan(...)` by default, but
the same pipeline is exposed as standalone methods on `MotionPlanner`
— handy when you want to:

- **Plan once with raw output** (`simplify=False, interpolate=False`)
  and apply them later.
- **Re-densify an old path** at a different `resolution` without
  replanning.
- **Keep the search raw, smooth only at display time** — the
  controller consumes the original waypoints, the visualiser gets
  a densely interpolated copy.
- **Drop simplification for constrained / cost plans** (which you
  want) but still apply it manually to specific segments where the
  shortcutting is known-safe.

```python
planner = create_planner(
    "wheelchair_arm",
    config=PlannerConfig(simplify=False, interpolate=False),
)
result = planner.plan(start, goal)                     # unsimplified, raw
smooth = planner.simplify_path(result.path, time_limit=1.0)
dense  = planner.interpolate_path(smooth, resolution=128.0)     # or count=200, resolution=0.0
```

Both methods reuse the planner's collision environment and
constraint set. `simplify_path` only consults the motion validator
(geometric shortcuts — not cost-aware, same caveat as the in-plan
flag). `interpolate_path` runs `StateSpace::interpolate` on the
existing edges, so it stays on the constraint manifold for projected
state spaces and does not perform collision checks itself.

## Standalone collision checking

The same VAMP SIMD pipeline that the planner drives per motion edge is
exposed directly — useful for filtering sampled goals, validating a
trajectory produced outside the planner, or scoring a whole candidate
batch of configurations at once.

```python
planner.validate(cfg)                   # single config → bool
planner.validate_batch(cfgs)            # (N, ndof) → (N,) bool array
```

`validate_batch` packs up to `rake` distinct configurations directly
into a VAMP `ConfigurationBlock<rake>` and runs **one** `fkcc<rake>`
call per block — the same primitive the motion validator uses for
interpolated samples along an edge, fed independent configs per lane
instead. Per-config result is preserved: when a packed block fails,
only that block falls back to per-lane checks.

A block of valid configurations costs one SIMD sweep; a block with an
invalid lane pays the per-lane fallback.

```python
# Example: filter 1000 candidate goals by validity.
goals = np.random.uniform(lo, hi, size=(1000, planner.num_dof))
mask = planner.validate_batch(goals)
good_goals = goals[mask]
```

## More

<div class="grid cards" markdown>

-   [__Subgroup planning__](subgroup.md)

    ---

    Plan over a slice of the 10-DOF body — the arm, the base, or the
    whole body — with the remaining joints pinned to any 10-DOF stance.

-   [__Manifold planning__](manifold.md)

    ---

    Hard task-space equality constraints written as CasADi
    expressions (planes, rails, orientation locks, couplings).
    Compiled once, cached, and injected into OMPL's
    `ProjectedStateSpace`.

-   [__Cost-space planning__](cost.md)

    ---

    Soft path-integral costs — the same CasADi authoring model, but
    now the constraint becomes a preference. Drives OMPL's
    asymptotically-optimal planners.

-   [__Time parameterization__](trajectory.md)

    ---

    Convert geometric paths into time-optimal trajectories with
    per-joint velocity and acceleration limits, using TOPP-RA.

-   [__Kinodynamic planning__](kinodynamic.md)

    ---

    FLASK: plan a time-parameterised trajectory directly, with joint
    and base velocity / acceleration limits and an exactly
    nonholonomic (diff-drive) base. Track it with feed-forward + PID.

</div>
