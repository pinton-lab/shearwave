# Reproducing the manuscript figures

This documents how each figure in `paper/figures/` is generated. Figures split
into two groups by what they require.

## Group 1 — reproducible from `shearwave` alone (no acoustic solver)

These need only this repository (`pip install -e ".[viz]"`), no `fullwave25`,
and no external acoustic data.

| Figure | Script | Data source |
|---|---|---|
| `fig_qiba_validation.png` | `paper/generate_fig_qiba.py` | `validation/qiba_run/qiba_summary.npz` |
| `shear_large_displacement.png`, `shear_slices_*.png` | `validation/validate_wavespeed.py` | self-contained |
| `convergence_*.png` | `validation/validate_convergence.py` | self-contained |
| `kelvin_voigt_1d_*.png` | `validation/validate_kelvin_voigt.py` | self-contained |

### QIBA figure

```bash
# 1. regenerate the phantom summary (shear sims on shearwave; ~tens of minutes)
python validation/validate_qiba.py                 # writes validation/qiba_run/qiba_summary.npz
# (optional) compare against the RSNA QIBA reference displacements:
#   python validation/validate_qiba.py --qiba-dir /path/to/qiba/res_sim_data
# 2. render the figure
python paper/generate_fig_qiba.py                  # writes paper/figures/fig_qiba_validation.png
```

The ARF push for the QIBA figure is built analytically in the script
(`_build_gaussian_arf`); the kymograph panel re-runs a shear simulation via
`from shearwave import shear_fdtd_staggered`. No acoustic (fullwave) input is
involved.

### Grid-convergence figure

The convergence sweep uses a **fixed physical source width** so every grid
discretizes the same continuum problem (see manuscript §4.2):

```bash
python validation/validate_convergence.py \
    --levels 60,90,120,150,180 --sigma-phys 0.003 --damp-mode constant-total \
    --out-dir paper/figures/convergence_run
cp paper/figures/convergence_run/convergence_trace_l2.png paper/figures/
```

Levels are multiples of 30 so the fixed physical sensor distances
(3.6/6.0/8.4/10.8 mm on a 36 mm domain) land exactly on grid nodes at every
level. `--sigma-phys` holds the Gaussian source at a fixed physical width
(overriding the legacy `--sigma-xy` grid-point width); `--damp-mode
constant-total` scales the per-step velocity damping so the cumulative damping
over the time window is grid-independent.

## Group 2 — require the acoustic solver (`fullwave25`)

The transcranial figures post-process the output of a GPU acoustic FDTD
simulation that this repository does **not** contain. `fullwave25` is an
optional dependency (`pip install -e ".[fullwave]"`); the acoustic solve
additionally needs a CUDA GPU.

| Figure | Script | Needs |
|---|---|---|
| `fig_bc_acoustical_and_pressure.png` | `generate_fig_bc.py` | sparse-array acoustic run |
| `fig_de_*`, `fig_de_arf_and_displacement.png` | `generate_fig_de.py` | sparse-array run (shear only) |
| `fig_tips_*` | `generate_fig_tips_*.py` | TIPS acoustic run |

### Regeneration workflow

The acoustic→shear coupling is: the acoustic solve produces a pressure field,
`shearwave.compute_radiation_force` converts it to an ARF body force
(`radiation_force.npy`), and `shearwave.shear_fdtd_staggered` propagates the
shear response. Everything downstream of the pressure field already lives in
`shearwave`.

```bash
# 1. run the end-to-end example (acoustic on GPU + shear on shearwave):
python examples/tips_transcranial_shear_3d.py        # writes outputs/tips_transcranial_shear/
python examples/sparse_transcranial_shear_3d.py      # writes outputs/sparse_transcranial_shear/
# 2. point the figure scripts at those output directories, then render.
```

> Note: the `generate_fig_tips_*.py` / `generate_fig_bc.py` / `generate_fig_de.py`
> scripts currently contain absolute input/output paths
> (`/home/gfp/fullwave25-private/...`) from the original prototype. Before
> rerunning, set their `data_dir` to the example output directory above and
> `out_dir` to `paper/figures/`. Only the two `examples/*_3d.py` generators
> require `fullwave25`; the figure scripts themselves are pure post-processing.
