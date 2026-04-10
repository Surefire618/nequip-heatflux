# nequip-heatflux

Heat flux calculator for [NequIP](https://github.com/mir-group/nequip).

This project is an adaptation of [gknet](https://github.com/sirmarcel/gknet-archive) (by [Marcel Langer](https://github.com/sirmarcel)) to NequIP.

## Compatibility

The current version targets the **legacy NequIP codebase up to and including version 0.6.1**. It relies on internal APIs from that release line (`nequip.scripts.deploy`, `nequip.train.Trainer`, `nequip.nn._grad_output`, `nequip.ase.NequIPCalculator`, ...) and has not been ported to the newer NequIP 0.7+ / `nequip-allegro` refactors.

## Installation

```
pip install -e .
```

You additionally need a working `nequip <= 0.6.1` install and PyTorch. For the Green-Kubo workflow below you also need [FHI-vibes](https://vibes-developers.gitlab.io/vibes/).

## Usage

The package exposes an ASE-compatible calculator factory:

```python
from nequip_calculator import calculator

calc = calculator(
    model_file="nequip/best_model.pth",
    device="cuda",
    calculator_name="unfolded",   # hardy | virial | unfolded | folded | nequip
)
atoms.calc = calc
```

`calculator_name` selects the heat-flux strategy:

- `nequip` - plain upstream `NequIPCalculator` (no heat flux).
- `hardy` - Hardy-virial per-atom stress, heat flux from `virials @ v`.
- `virial` - per-atom stress from strain Jacobian.
- `unfolded` - two-pass calculator that unfolds the periodic cell up to the effective interaction cutoff and computes the heat flux via autograd on per-atom energies. This is the recommended path.
- `folded` - experimental single-cell variant of `unfolded`.

## Example: FHI-vibes Green-Kubo

A ready-to-run FHI-vibes example is provided in `example/vibes/` (KI bcc, 5x5x5 supercell, NequIP model included). From inside that directory:

```
make run   # vibes run md         -- run the MD using the unfolded heat-flux calculator
make md    # vibes output md ...  -- post-process the trajectory
make gk    # vibes output gk ...  -- Green-Kubo post-processing
```

The corresponding `md.in` wires this package into vibes as:

```
[calculator]
module: nequip_calculator
name:   calculator

[calculator.parameters]
model_file:      nequip/best_model.pth
device:          cuda
calculator_name: unfolded
```

## Citation

If you use this package in published work, please cite the two papers that introduced and justified the heat-flux formulations implemented here:

- M. F. Langer, F. Knoop, C. Carbogno, M. Scheffler, and M. Rupp, *Heat flux for semilocal machine-learning potentials*, Physical Review B **108**, L100302 (2023).

  ```bibtex
  @article{langer2023heatflux,
    title   = {Heat flux for semilocal machine-learning potentials},
    author  = {Langer, Marcel F. and Knoop, Florian and Carbogno, Christian and Scheffler, Matthias and Rupp, Matthias},
    journal = {Physical Review B},
    volume  = {108},
    number  = {10},
    pages   = {L100302},
    year    = {2023},
  }
  ```

- M. F. Langer, J. T. Frank, and F. Knoop, *Stress and heat flux via automatic differentiation*, The Journal of Chemical Physics **159**, 174105 (2023).

  ```bibtex
  @article{langer2023stressheatflux,
    title   = {Stress and heat flux via automatic differentiation},
    author  = {Langer, Marcel F. and Frank, J. Thorben and Knoop, Florian},
    journal = {The Journal of Chemical Physics},
    volume  = {159},
    number  = {17},
    year    = {2023},
  }
  ```

Please also cite [NequIP](https://github.com/mir-group/nequip) and [FHI-vibes](https://vibes-developers.gitlab.io/vibes/) as appropriate for your workflow.
