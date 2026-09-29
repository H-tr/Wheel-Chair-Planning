# Getting Started

## Prerequisites

- **Linux** x86_64 with AVX2 (Haswell / 2013 or newer)
- **Python** 3.12–3.14

## Installation

Pre-built wheels are available for Python 3.12–3.14 on Linux x86_64. No
local compilation required:

```bash
pip install wheelchair-planning
```

The core install pulls in only `numpy`, `scipy` and `casadi`: motion
planning (OMPL + VAMP), FLASK kinodynamic planning, TOPP-RA time
parameterization and constraint / cost planning all work out of the box.
Optional extras:

| Extra | Adds | Enables |
|---|---|---|
| `kinematics` | `pin`, `pin-pink`, `osqp` | Pinocchio FK, collision model, Pink IK |
| `examples` | `pybullet`, `fire`, `matplotlib`, `xmltodict` | PyBullet environments and the example scripts |
| `dev` | `pytest`, `trimesh`, `pre-commit` | The test suite |

```bash
pip install "wheelchair-planning[kinematics,examples]"
```

The TRAC-IK solver needs the compiled `pytracik` extension, which is only
built when orocos-kdl and NLopt are available — use the pixi development
environment below for it.

## Verify installation

```python
from wheelchair_planning.planning import create_planner
from wheelchair_planning.wheelchair import HOME_JOINTS

planner = create_planner("wheelchair_arm")
start = planner.extract_config(HOME_JOINTS)
result = planner.plan(start, planner.sample_valid())
print(f"Planning {'succeeded' if result.success else 'failed'}")
```

## Building Wheels from Source

Release wheels are built with [cibuildwheel](https://cibuildwheel.pypa.io)
using the configuration in `pyproject.toml`:

```bash
pipx run cibuildwheel --platform linux
```

This builds manylinux x86_64 wheels for every supported Python version
into `wheelhouse/`. Docker must be installed and running.

## Development Setup

For contributing or rebuilding C++ dependencies from source, use [pixi](https://pixi.sh):

```bash
git clone --recursive https://github.com/H-tr/Wheel-Chair-Planning.git
cd Wheel-Chair-Planning
bash scripts/setup.sh
```

Or manually:

```bash
git clone --recursive https://github.com/H-tr/Wheel-Chair-Planning.git
cd Wheel-Chair-Planning
pixi install
pixi run cricket-build
pixi run foam-build
bash scripts/download_assets.sh
```
