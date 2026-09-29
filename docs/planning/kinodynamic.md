# Kinodynamic Planning (FLASK)

`MotionPlanner.plan_kinodynamic` plans a **time-parameterised,
dynamically feasible trajectory** directly — no separate time
parameterization step. It implements FLASK (Duong et al., "Ultrafast
Sampling-based Kinodynamic Planning via Differential Flatness", T-RO
2026): RRT-Connect in the differentially flat output space, where every
tree edge is a closed-form cubic that is collision-checked with the same
SIMD VAMP kernels as the geometric planner. The implementation is ported
from [Fetch-Planning](https://github.com/H-tr/Fetch-Planning).

```python
from wheelchair_planning.planning import create_planner
from wheelchair_planning.types import KinodynamicConfig

planner = create_planner("wheelchair_whole_body", pointcloud=cloud)
planner.set_base_bounds(-2.0, 3.0, -1.5, 1.5)   # sample the base near the task
result = planner.plan_kinodynamic(start, goal, config=KinodynamicConfig(time_limit=1.0))

traj = result.trajectory                      # KinodynamicTrajectory
times, q, qd, qdd = traj.sample_uniform(dt=0.01)
twist = traj.sample_base_twist(times)         # (T, 2): v, omega for the base
```

It works for every subgroup (`wheelchair_arm`, `wheelchair_base`,
`wheelchair_whole_body`). See `examples/planning/kinodynamic.py`.

## The wheelchair base

The wheelchair is a differential drive: the rear hub-motor wheels drive
and the front omni wheels roll freely. `base_link`'s origin — the planned
`(x, y)` — is the midpoint of the rear drive axle, so the base moves
along its heading or rotates in place, never sideways. The planner's
output satisfies this exactly (zero lateral velocity).

The seated user faces +x, the omni-wheel end. By default the chair drives
forward only and turns in place where needed, so it always moves the way
its user faces (`BASE_REVERSE_ENABLE = False`; pass
`KinodynamicConfig(allow_reverse=True)` to permit reversing).

Base positions are sampled uniformly inside the planner's base bounds,
which default to the URDF's ±10 m virtual-joint limits. Call
`set_base_bounds(x_lo, x_hi, y_lo, y_hi)` around the task's workspace;
the bounds persist across `set_subgroup` and also apply to the geometric
planners.

## How it works

<div class="grid cards" markdown>

-   __Flat outputs__

    ---

    Arm joints are their own flat outputs (fully actuated). The
    diff-drive base contributes its planar position `(x, y)`; the
    heading follows the direction of travel,
    `theta = atan2(y_dot, x_dot) + gear * pi`.

-   __Closed-form edges__

    ---

    Nodes are `(position, velocity)` states. An edge is the cubic
    Hermite between its endpoints — the minimum-effort (LQMT) motion —
    with the duration `T*` that minimises
    `sum_i int y_i''^2 / a_max_i^2 dt + rho T` (a quartic root).

-   __Limits__

    ---

    Joint position / velocity / acceleration bounds are checked exactly
    on each cubic; `T*` is stretched until they and the base speed,
    acceleration and yaw rate hold. Base speed, tangential acceleration
    and yaw rate are then checked on a 50 ms grid.

-   __SIMD collision checks__

    ---

    Each edge is sampled at VAMP's resolution along its arc length and
    checked in interleaved `ConfigurationBlock` batches (FLASK Alg. 3),
    then the solution is simplified the way VAMP simplifies geometric
    paths: greedy shortcuts (Alg. 4) plus subdivide-and-cut-corners
    smoothing, keeping a change only if it lowers the cost.

</div>

### The base at rest

The unicycle model has no heading at zero velocity — exactly where a
mobile manipulator starts, stops and grasps. The planner makes parked
states first-class:

* A **parked node** keeps its heading. Edges leave and enter it along
  that heading (forward or reverse); when a tree grows out of a parked
  node, the sampled target's lateral velocity is projected so the cubic
  departs aligned.
* **Gears** only change at parked nodes, so every drive segment has a
  well-defined heading and no hidden cusp.
* **Rotate in place**: a connection into or out of a parked node whose
  heading does not match gets a spin segment (arm holding still).
  Between two parked nodes that is rotate–translate–rotate, which is why
  open-floor legs solve in about a millisecond.

## Tracking the trajectory

The trajectory is C¹ in every flat output: positions and velocities are
continuous; accelerations — and the base yaw rate, which depends on
them — jump at segment knots. A feed-forward + feedback controller is
enough:

* **Joints**: `u = qd_ref + Kp (q_ref - q) + Ki ∫(q_ref - q)` as a
  velocity command (what a joint trajectory controller does).
* **Base**: feed-forward `(v, omega)` from `traj.base_twist(t)` plus
  pose feedback in the body frame (e.g. a Kanayama law).

Pass the current state to replan on the fly — the new trajectory starts
with the given velocity:

```python
q_now, qd_now = traj.position(t), traj.velocity(t)
result = planner.plan_kinodynamic(q_now, new_goal, start_velocity=qd_now)
```

!!! warning "Clearance"

    Trajectories are collision-free at zero clearance, up to the sampling
    resolution like every VAMP edge. Tracking error is not accounted for:
    inflate obstacles with `PlannerConfig.point_radius` for a margin, and
    replan from the measured state.

## Configuration

`KinodynamicConfig` holds the planner parameters. The limits come from
`wheelchair_planning/wheelchair.py` (`JOINT_VELOCITY_LIMITS`,
`JOINT_ACCELERATION_LIMITS`, `BASE_MAX_*`, `BASE_REVERSE_ENABLE`); they
are placeholders (base 0.6 m/s, 1.0 rad/s, forward only) — tune them to
the robot's controllers. `velocity_scale` / `acceleration_scale` scale
all of them at once.

| Field | Default | Meaning |
|---|---|---|
| `time_limit` | `1.0` | Planning budget (s) |
| `rho` | `1.0` | Time weight; rest-to-rest edges peak at `sqrt(rho)` × the acceleration limit |
| `max_extension_time` | `3.0` | Tree step: longer cubics are truncated (node stays on the cubic) |
| `rest_sample_probability` | `0.3` | Share of zero-velocity samples (straight-line edges, gear changes) |
| `spin_probability` | `0.3` | Share of rotate-in-place extensions from parked nodes |
| `allow_reverse` | `BASE_REVERSE_ENABLE` | Permit the reverse gear |
| `simplify_iterations` / `simplify_time_limit` | `5` / `0.01` | Rounds and time cap (s) for the VAMP-style simplification; raise the cap for shorter trajectories |
| `seed` | `0` | RNG seed (`0` = nondeterministic) |

## Results

Planning time per call (search + simplification) with the default limits
(base 0.6 m/s, 1.0 rad/s, forward only), 3 s budget, 20 seeds per leg,
measured on a shared desktop CPU under load, so absolute times are
pessimistic. Every trajectory was re-checked independently — dense
collision resampling every 2 ms, joint and base limits, C¹ continuity,
zero lateral velocity:

| Leg | Solved | Median | p90 |
|---|---|---|---|
| arm only: around a shelf | 20/20 | 1.8 ms | 2.5 ms |
| whole body: rotate in place + move the arm | 20/20 | 0.2 ms | 0.2 ms |
| whole body: sideways and turn around | 20/20 | 0.4 ms | 0.4 ms |
| whole body: behind, turn and drive | 20/20 | 0.5 ms | 0.6 ms |
| whole body: 2.5 m ahead, pillar in the way | 20/20 | 24 ms | 103 ms |
| base only: drive to the table around a pillar | 20/20 | 25 ms | 32 ms |
| whole body: drive to the table and reach | 20/20 | 98 ms | 158 ms |

One of the 140 trajectories grazed an obstacle between the planner's own
collision samples: the tucked gripper touched a pillar point by 0.5 mm
over ~1 cm of travel. Edges are checked at VAMP's resolution (~1.4 cm
here), like the geometric planner's, so keep `PlannerConfig.point_radius`
as the clearance margin.

The base limits dominate planning time: with 0.3 m/s and 0.5 rad/s most
curved edges are infeasible, and (measured with reverse allowed) the drive
to the table took about twice as long to plan and the whole-body legs about
three times as long.

Two things keep the collision checks cheap. The chassis spheres are spread
over twelve fixed child links (`scripts/split_collision_link.py`), so the
checker's per-link bounding spheres skip the parts of the chassis the arm
is nowhere near; one check costs ~1 µs per configuration in a SIMD batch.
And durations are stretched for the base yaw-rate limit, not only for
speed and acceleration, so an edge that turns too fast is slowed down
instead of rejected.

Driving with the arm tucked and then moving the arm is the robust
pattern; planning base and arm together is an 18-dimensional flat-state
problem and needs a longer budget in cluttered scenes.
