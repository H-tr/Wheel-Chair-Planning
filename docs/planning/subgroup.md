# Subgroup Planning

The wheelchair robot has 10 planning joints: the planar base
(`Joint_Virtual_X`, `Joint_Virtual_Y`, `Joint_Virtual_Theta`) and the
xArm7 (`joint1` … `joint7`). Often only a slice of the body is under
active control while the rest stays put.

**Subgroups are user-defined.** A subgroup is an ordered list of
indices into the 10-DOF body. The OMPL planner reasons in the reduced
joint space; the C++ collision pipeline injects the frozen stance you
choose and still checks the **full** body.

<video controls loop muted playsinline width="100%">
  <source src="../../assets/subgroup_planning.mp4" type="video/mp4">
</video>

## Built-in aliases

Aliases ship for the common slicings, discoverable via
[`available_robots()`](../api/planning.md#wheelchair_planning.planning.motion_planner.available_robots):

| Alias | DOF | What it controls |
|---|---:|---|
| `wheelchair_arm` | 7 | xArm7 arm, base held fixed |
| `wheelchair_base` | 3 | mobile base (x, y, yaw) |
| `wheelchair_whole_body` | 10 | base + arm |
| `wheelchair` | 10 | full body |

The list is a dict in `wheelchair_planning.wheelchair.PLANNING_SUBGROUPS`
— add your own entry to extend it, or use the low-level API below for
one-off slicings.

## Using a built-in alias

```python
from wheelchair_planning.wheelchair import HOME_JOINTS
from wheelchair_planning.planning import create_planner
from wheelchair_planning.types import PlannerConfig

live_config = robot.get_joint_state()          # 10-D array from your system

planner = create_planner(
    "wheelchair_arm",
    config=PlannerConfig(time_limit=1.0),
    base_config=live_config,                    # frozen joints anchored here
    pointcloud=obstacle_cloud,
)

start = planner.extract_config(live_config)     # 10-D → 7-D
goal = planner.sample_valid()
result = planner.plan(start, goal)

full_path = planner.embed_path(result.path)     # 7-D path → 10-D path
```

`base_config` defaults to `HOME_JOINTS`. Pass the live joint reading
instead to pin every non-active joint exactly where the robot
currently is — collision checks see the real stance, not a synthetic
home pose. `planner.set_subgroup(robot_name, base_config=None)` switches
an existing planner to another alias without rebuilding the collision
environment; joint limits set with `set_joint_limits` are kept.

## Defining a custom subgroup

Any ordered list of joint indices is a valid subgroup. Drop to the C++
binding when the alias list doesn't match your problem:

```python
from wheelchair_planning._ompl_vamp import OmplVampPlanner
from wheelchair_planning.wheelchair import HOME_JOINTS, wheelchair_robot_config

# Name the joints you want to plan over; the rest stay frozen.
active_names = ["joint1", "joint2", "joint3", "joint4"]
full_names = wheelchair_robot_config.joint_names
active_indices = [full_names.index(j) for j in active_names]

planner = OmplVampPlanner(active_indices, HOME_JOINTS.tolist())
planner.add_pointcloud(cloud.tolist(), *planner.min_max_radii(), 0.012)

start = [HOME_JOINTS[i] for i in active_indices]
goal = [...]                                    # your own sampling logic
result = planner.plan(start, goal, planner_name="rrtc", time_limit=1.0)
```

Same planner, same collision pipeline, same point-cloud broadphase
— just a different slice. You can also switch an existing planner
to a different subgroup on the fly with
[`set_subgroup(active_indices, frozen_config)`](../api/planning.md#wheelchair_planning._ompl_vamp.OmplVampPlanner.set_subgroup),
which preserves the collision environment.

## When it matters

Planning the whole body for a task that only needs the arm searches
base configurations that do not matter. Whole-body subgroups are the
right choice when the base has to move with the arm. Pick the smallest
subgroup that captures the joints your task needs to move; the
[benchmarks](index.md#benchmarks) give each subgroup's planning times.
