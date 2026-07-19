# shearwave

A collocated (cell-centered) finite-difference time-domain (FDTD) solver for shear wave propagation in viscoelastic media with Kelvin-Voigt damping.

## Installation

```bash
pip install -e .
```

For JAX GPU acceleration:
```bash
pip install -e ".[jax-gpu]"
```

## Quick start

```python
import numpy as np
from shearwave import shear_fdtd

# Define body force, material properties, grid spacing, time step
u, v = shear_fdtd(bx, by, bz, rho, mu, dX, dY, dZ, dT, nT, opts={})
```

## Features

- 3D collocated (cell-centered) FDTD with central-difference time stepping
- Kelvin-Voigt viscoelastic damping
- Helmholtz-Hodge decomposition for divergence-free force projection
- Conjugate-gradient Poisson solver
- NumPy (CPU) and JAX (GPU/TPU) backends
- Acoustic radiation force computation (plane-wave and Poynting vector)
- Strain tensor and invariant computation
- Acoustic strain and strain gradient analysis

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.
