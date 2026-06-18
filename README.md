# Wheel-Chair-Planning

<div align="center">

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue?style=for-the-badge&logo=materialformkdocs)](https://h-tr.github.io/Wheel-Chair-Planning/)
[![Build Docs](https://img.shields.io/github/actions/workflow/status/H-tr/Wheel-Chair-Planning/docs.yml?branch=main&style=for-the-badge&label=docs%20build&logo=github)](https://github.com/H-tr/Wheel-Chair-Planning/actions/workflows/docs.yml)
[![CI](https://img.shields.io/github/actions/workflow/status/H-tr/Wheel-Chair-Planning/ci.yml?branch=main&style=for-the-badge&label=CI&logo=github)](https://github.com/H-tr/Wheel-Chair-Planning/actions/workflows/ci.yml)

</div>

A planning library for the **wheelchair-mounted xArm7** robot
([`wheelchair_xarm_description`](https://github.com/soibkhon/wheelchair_xarm_description)).
It provides inverse kinematics (TRAC-IK and Pink), motion planning
([VAMP](https://github.com/KavrakiLab/vamp) + [OMPL](https://ompl.kavrakilab.org/)),
and collision-aware planning through a unified Python interface.

The architecture mirrors [AdaCompNUS/Autolife-Planning](https://github.com/AdaCompNUS/Autolife-Planning);
only the robot has been swapped — from the 24-DOF Autolife humanoid to a 10-DOF
mobile manipulator (a 3-DOF planar wheelchair base + a 7-DOF xArm7 arm).

## Features

- **Inverse Kinematics** — TRAC-IK (unconstrained) and Pink (QP-based constrained) solvers with self-collision avoidance and camera/orientation stabilization
- **Motion Planning** — VAMP-based planner with collision checking, path validation, and subgroup planning (arm-only, base-only, and whole-body)
- **Time Parameterization** — TOPP-RA by default, with legacy TOTG available; converts planned paths into executable trajectories with velocity/acceleration limits
- **Collision Geometry** — Spherized URDF representations for efficient collision detection, with pointcloud obstacle support

## Robot

| | |
|---|---|
| Description | [`soibkhon/wheelchair_xarm_description`](https://github.com/soibkhon/wheelchair_xarm_description) (vendored at `assets/wheelchair_xarm_description`) |
| Arm | UFACTORY xArm7 (7 DOF), end-effector `link_tcp` |
| Base | Wheelchair platform, modeled as a 3-DOF planar virtual joint (x, y, yaw) |
| Planning groups | `wheelchair_arm` (7), `wheelchair_base` (3), `wheelchair_whole_body` (10) |

## Quick Start

**Platform**: Linux, Python 3.11+ (see `pixi.toml`).

For inference — running the planners and IK solvers — pip install:

```bash
git clone --recursive https://github.com/H-tr/Wheel-Chair-Planning.git
cd Wheel-Chair-Planning
pip install -e .
```

For development — rebuilding URDFs, regenerating FK headers, running the C++
toolchain end-to-end — use the setup script, which installs pixi and the
conda-forge deps (pinocchio, orocos-kdl, eigen, boost, ...):

```bash
bash scripts/setup.sh
```

## Usage

```bash
# Inverse kinematics
pixi run python examples/ik/basic.py
pixi run -e dev python examples/ik/basic_vis.py           # PyBullet visualization

# Motion planning
pixi run python examples/planning/motion.py
pixi run python examples/planning/subgroup.py

# Time parameterization
pixi run python examples/planning/time_parameterization.py

# Tests
pixi run -e dev test
```

### Rebuilding the robot description

The planning URDFs, spherized collision geometry, and the FK header are generated
from the vendored robot description:

```bash
pixi run build-robot        # preprocess URDF + meshes + SRDF + virtual base
pixi run decompose-robot    # CoACD convex decomposition
pixi run spherize-robot     # FOAM spherization -> wheelchair_spherized.urdf
pixi run generate-fk        # cricket -> ext/ompl_vamp/robot/wheelchair.hh
pixi run build-pkg          # compile the C++ extensions
```

## Project Structure

```
wheelchair_planning/   # Core Python package
  kinematics/          # TRAC-IK + Pink IK, FK, collision checking
  planning/            # VAMP motion planning, cost + constrained planners
  trajectory/          # TOPP-RA / TOTG time parameterization
  envs/                # Simulation environments (PyBullet)
  types/               # Shared dataclasses (Pose, JointState, ...)
  resources/           # Packaged URDFs and asset loaders
third_party/
  vamp/                # SIMD-accelerated motion planning (submodule)
  ompl/                # Open Motion Planning Library (submodule)
  cricket/             # FK code generator (submodule)
  foam/                # Collision geometry / spherization (submodule)
  toppra/              # Time-optimal path parameterization (submodule)
ext/                   # C++ extensions (ompl_vamp, trac_ik, time_parameterization)
assets/
  wheelchair_xarm_description/   # Raw robot description (submodule)
examples/              # IK, planning, demos
tests/                 # Pytest suite (CI)
scripts/               # Setup, build, spherize, FK codegen
docs/                  # MkDocs site sources (GitHub Pages)
```

## Acknowledgements

This project builds on several outstanding open-source libraries and on the
[AdaCompNUS/Autolife-Planning](https://github.com/AdaCompNUS/Autolife-Planning)
architecture:

- **[VAMP](https://github.com/KavrakiLab/vamp)** — SIMD-accelerated motion planning and collision checking (KavrakiLab, Rice University).
- **[OMPL](https://ompl.kavrakilab.org/)** — The Open Motion Planning Library (KavrakiLab, Rice University).
- **[TOPP-RA](https://github.com/hungpham2511/toppra)** — Reachability-analysis-based time-optimal path parameterization, used as the default timing backend.
- **[MoveIt 2](https://github.com/moveit/moveit2)** — The vendored TOTG implementation in `ext/time_parameterization/` is adapted from MoveIt 2's `trajectory_processing` module (Tobias Kunz and Mike Stilman, Georgia Tech). See `ext/time_parameterization/LICENSE.TOTG`.
- **[TRAC-IK](https://traclabs.com/projects/trac-ik/)** — Inverse kinematics solver (TRACLabs).
- **[wheelchair_xarm_description](https://github.com/soibkhon/wheelchair_xarm_description)** — The robot description (URDF + meshes).
```
