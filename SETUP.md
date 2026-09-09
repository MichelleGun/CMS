# Environment setup

Done on Windows 11, ASUS Zenbook 14 (no NVIDIA GPU → PyTorch runs on CPU).
Adapt paths for other machines.

## Prerequisites

- **Miniconda / Anaconda**
- **Git**
- **A C++ compiler** — `pybullet` has no pre-built wheel for Python > 3.10, so pip
  compiles it from source.
  - Windows: Visual Studio Build Tools with the "Desktop development with C++" workload
  - Ubuntu: `sudo apt install build-essential`
  - macOS: Xcode command-line tools

## Steps

```bash
# 1. Get the simulator
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones

# 2. Create the environment.
#    gym-pybullet-drones 2.2.0 requires Python >= 3.12.
#    Using the conda-forge channel avoids the Anaconda Terms-of-Service prompt.
conda create -n drones -c conda-forge --override-channels python=3.12 -y
conda activate drones

# 3. conda-forge's python=3.12 ships without pip — add it
conda install -c conda-forge --override-channels pip setuptools wheel -y

# 4. Install the package + all deps (compiles pybullet, downloads torch ~124 MB;
#    allow 10-20 min)
pip install -e .
```

Verify:

```bash
python -c "import pybullet, torch, stable_baselines3, gymnasium; \
from gym_pybullet_drones.envs.HoverAviary import HoverAviary; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
e = HoverAviary(); e.reset(); print('env OK')"
```

Expected: `torch 2.14.0+cpu cuda False` … `env OK`.

## Exact versions used

See `requirements-lock.txt`. Key ones:

| package | version |
|---|---|
| python | 3.12 |
| torch | 2.14.0+cpu |
| stable-baselines3 | 2.9.0 |
| gymnasium | 1.3.0 |
| pybullet | 3.2.7 (built from source) |
| gym-pybullet-drones | 2.2.0 |

## Running the scripts in this folder

The scripts import `gym_pybullet_drones` as an installed package, so they run from
anywhere once `conda activate drones` is done. They read/write a `results_quickstart/`
folder relative to the current directory — `cd` into `scripts/` first, or pass an
explicit `--model_path` to the eval/play commands.
