# Inverse Kinematics

Wheelchair Planning ships two IK backends behind a single factory
([`create_ik_solver`](../api/kinematics.md#wheelchair_planning.kinematics.ik_solver_base.create_ik_solver)):

- **TRAC-IK** — fast numerical IK. Runs SQP and a pseudoinverse
  solver concurrently and returns the first valid solution. Best
  for unconstrained point-to-point targets.
- **Pink** — differential QP IK. Composes an end-effector task with
  any number of soft secondary objectives (CoM, posture, camera,
  self-collision). Best when the solution must satisfy extra
  task-space invariants.

Both backends cover the same kinematic chains:

| Chain | DOF | Span |
|---|---|---|
| `arm` | 7 | xArm7, `link_base` → `link_tcp` |
| `whole_body` | 10 | planar base + arm, `Link_Zero_Point` → `link_tcp` |

Pick the chain with `create_ik_solver("chain_name", ...)` — see the
`JOINT_GROUPS` table in `wheelchair_planning.wheelchair` for the mapping
into the full 10-DOF body configuration.

<div class="grid cards" markdown>

-   [__Unconstrained IK (TRAC-IK)__](unconstrained.md)

    ---

    One call per target. Fast, thread-safe, deterministic under a fixed
    seed. Use when you have a reachable pose and want a solution.

-   [__Constrained IK (Pink)__](constrained.md)

    ---

    QP solve per step. Compose the end-effector task with CoM
    stability, camera stabilization, self-collision avoidance, and
    posture regularization.

</div>
