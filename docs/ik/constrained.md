# Constrained IK (Pink)

Pink solves IK as a quadratic program at every time step. A primary
task — usually the end-effector pose — is composed with any number
of soft secondary objectives: centre-of-mass stability, camera-frame
stabilization, self-collision avoidance, and a posture regularizer.
Each step solves for the joint velocity that best satisfies all of
them under the given weights and integrates it over `dt`; the solve
returns the resulting joint positions.

<video controls loop muted playsinline width="100%">
  <source src="../../assets/constrained_ik.mp4" type="video/mp4">
</video>

## Minimal example

```python
import numpy as np
from wheelchair_planning.wheelchair import HOME_JOINTS
from wheelchair_planning.kinematics import create_ik_solver
from wheelchair_planning.types import PinkIKConfig, SE3Pose

solver = create_ik_solver(
    "whole_body",
    backend="pink",
    config=PinkIKConfig(
        dt=0.01,
        max_iterations=300,
        position_cost=1.0,
        orientation_cost=1.0,
        com_cost=0.1,              # penalize CoM drift from the seed
        lm_damping=1e-3,           # Levenberg-Marquardt damping
        posture_cost=1e-3,         # soft pull toward seed posture
    ),
)

home = HOME_JOINTS                  # whole_body spans the full 10-DOF config
ee = solver.fk(home)
target = SE3Pose(
    position=ee.position + np.array([0.15, -0.05, 0.20]),
    rotation=ee.rotation,
)

result = solver.solve(target, seed=home)
print(result.status, result.joint_positions)
```

## Secondary objectives

Every objective is a cost term on the step's QP objective; `*_cost`
fields tune the weight relative to the primary end-effector task.

| Field | Effect |
|---|---|
| `position_cost` / `orientation_cost` | weight of the primary EE task |
| `com_cost` | penalizes CoM drift from its position at the seed configuration |
| `camera_frame` + `camera_cost` | anchors the frame's full pose at its seed pose — e.g. the wrist camera `camera_color_optical_frame` |
| `posture_cost` | pulls unused joints toward the seed posture |
| `lm_damping` | Levenberg-Marquardt damping near singularities |
| `self_collision` | adds self-collision barriers; the solver also needs a collision model — pass `self_collision=True` to `create_ik_solver` (pairs filtered by the URDF's co-located SRDF, if any) or call `set_collision_context` |

## When to use

- **Whole-body reaching** with the mobile base included — TRAC-IK
  gives you a solution, Pink also holds the CoM near its seed position
  while you reach for it.
- **Camera-stable reaching** — anchor the wrist camera frame at its
  seed pose; it rides on the end effector, so `camera_cost` trades
  reach accuracy for a steadier view.
- **Seed-to-solution motion** — `solve_constrained` also returns the
  joint-position trajectory of the `dt` integration steps from the
  seed to the solution.

## API reference

See [Kinematics → PinkIKSolver](../api/kinematics.md#wheelchair_planning.kinematics.pink_ik_solver.PinkIKSolver).
