import contextlib

import numpy as np
import torch

from ase import units
from ase.calculators.calculator import Calculator, all_changes

from nequip.ase import NequIPCalculator
from nequip.data import AtomicData, AtomicDataDict

from .unfolder import Unfolder

# nequip wraps the energy model in these to add autograd-based outputs.
# Their forward() calls torch.autograd.grad internally without retain_graph,
# which frees the graph we need for the heat-flux autograd. We splice them
# out of the wrapper chain for the heat-flux forward pass, keeping any
# surrounding wrappers (RescaleOutput, GraphModel, ...) intact so that
# rescaling is still applied to per-atom energies.
_GRAD_WRAPPER_NAMES = (
    "StressOutput",
    "StressForceOutput",
    "GradientOutput",
    "ForceOutput",
    "PartialForceOutput",
    "StrainStressOutput",
)

# Fields that only exist after a grad-output wrapper has run. When we splice
# the grad wrapper out for the heat-flux forward pass, any ancestor
# RescaleOutput that still references these in its scale/shift key lists
# would KeyError because the inner energy model never produces them.
_GRAD_ONLY_FIELDS = frozenset(
    (
        AtomicDataDict.FORCE_KEY,
        AtomicDataDict.PARTIAL_FORCE_KEY,
        AtomicDataDict.STRESS_KEY,
        AtomicDataDict.VIRIAL_KEY,
    )
)


def _find_grad_wrapper_parent(module, ancestors=()):
    """Locate the first grad-output wrapper in a nequip model tree.

    nequip models are typically structured as e.g.
    ``GraphModel.model = RescaleOutput.model = GradientOutput.func = SequentialGraphNetwork``.
    We walk ``.model`` and ``.func`` attributes until we find a module whose
    class name is in ``_GRAD_WRAPPER_NAMES``, and return a dict with:
    - ``parent``: the module whose child is the wrapper
    - ``attr``:   attribute name on ``parent`` holding the wrapper
    - ``wrapper``: the wrapper module itself
    - ``rescale_ancestors``: list of RescaleOutput nodes encountered on the
      path from ``module`` down to ``wrapper`` (so we can temporarily patch
      their scale/shift key lists)

    Returns ``None`` if no wrapper is found.
    """
    for attr in ("model", "func"):
        sub = getattr(module, attr, None)
        if not isinstance(sub, torch.nn.Module):
            continue
        next_ancestors = ancestors
        if type(sub).__name__ == "RescaleOutput":
            next_ancestors = ancestors + (sub,)
        if type(sub).__name__ in _GRAD_WRAPPER_NAMES and hasattr(sub, "func"):
            return {
                "parent": module,
                "attr": attr,
                "wrapper": sub,
                "rescale_ancestors": list(ancestors),
            }
        found = _find_grad_wrapper_parent(sub, next_ancestors)
        if found is not None:
            return found
    return None


@contextlib.contextmanager
def _spliced_grad_wrapper(location):
    """Temporarily splice a grad-output wrapper out of the nequip model chain.

    For the duration of the ``with`` block:
    - ``location['parent'].<attr>`` is rebound to ``wrapper.func`` so the
      outer chain bypasses the grad wrapper entirely.
    - Any ``RescaleOutput`` ancestor's ``scale_keys`` / ``shift_keys`` lists
      are filtered to remove fields that only the grad wrapper produces
      (forces, stress, virial, partial forces). Without this,
      ``RescaleOutput.forward`` would ``KeyError`` trying to rescale a
      missing ``forces`` tensor.

    Everything is restored on exit. If ``location`` is ``None``, no-op.
    """
    if location is None:
        yield
        return

    parent = location["parent"]
    attr = location["attr"]
    wrapper = location["wrapper"]
    rescales = location["rescale_ancestors"]

    saved_keys = []
    for r in rescales:
        saved_keys.append(
            (r, list(r.scale_keys), list(r.shift_keys), list(r._all_keys))
        )
        r.scale_keys = [k for k in r.scale_keys if k not in _GRAD_ONLY_FIELDS]
        r.shift_keys = [k for k in r.shift_keys if k not in _GRAD_ONLY_FIELDS]
        r._all_keys  = [k for k in r._all_keys  if k not in _GRAD_ONLY_FIELDS]

    setattr(parent, attr, wrapper.func)
    try:
        yield
    finally:
        setattr(parent, attr, wrapper)
        for r, sk, shk, ak in saved_keys:
            r.scale_keys = sk
            r.shift_keys = shk
            r._all_keys  = ak


class UnfoldedHeatFluxCalculator(NequIPCalculator):
    def __init__(
        self,
        skin=None,
        skin_unfolder=0.1,
        n_interactions=1.,
        report_update=False,
        never_update=False,
        *args,
        **kwargs
    ):
        NequIPCalculator.__init__(self, *args, **kwargs)

        self._grad_wrapper_location = _find_grad_wrapper_parent(self.model)
        if self._grad_wrapper_location is None:
            print(
                "UnfoldedHeatFluxCalculator: no GradientOutput/StressOutput "
                "wrapper found on the loaded model; the heat-flux autograd "
                "may fail if the model frees its graph internally."
            )

        # effective cutoff
        cutoff = self.r_max
        effective_cutoff = cutoff * n_interactions
        self.effective_cutoff = effective_cutoff

        if skin is None:
            # unfolder allows movement up to skin/2 in *each direction*,
            # so we pick a skin that corresponds to the 3D distance
            skin = np.sqrt(3) * skin_unfolder

        self.unfolder = Unfolder(
            effective_cutoff,
            skin=skin_unfolder,
            report_update=report_update,
            never_update=never_update,
        )


    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):

        n = len(atoms)
        volume = atoms.get_volume()
        unfolded = self.unfolder(atoms)

        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data for calculating energy, forces, stress
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]
        data = self.transform(data)
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        # calculate energy, forces, stress
        data = self.model(data)

        self.results = {}
        self.results["forces"] = data[AtomicDataDict.FORCE_KEY][:n,:].detach().cpu().numpy()
        if AtomicDataDict.STRESS_KEY in data:
            self.results["stress"] = data[AtomicDataDict.STRESS_KEY][:n,:,:].detach().squeeze().cpu().numpy()
        self.results["energies"] = data[AtomicDataDict.PER_ATOM_ENERGY_KEY][:n, :].detach().cpu().numpy()
        self.results["energy"] = self.results["energies"].sum()

        del data # do we need this ?

        # prepare data
        data = AtomicData.from_ase(atoms=unfolded.atoms, r_max=self.r_max)
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]
        data = self.transform(data)
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        # predict + extract data
        pos = data[AtomicDataDict.POSITIONS_KEY]

        velocities = torch.tensor(unfolded.atoms.get_velocities() * units.fs).to(self.device)
        aux_pos = pos.detach().squeeze()[:n, :]

        pos.requires_grad_(True)
        with _spliced_grad_wrapper(self._grad_wrapper_location):
            data = self.model(data)
        energies = data[AtomicDataDict.PER_ATOM_ENERGY_KEY][:n, :]

        potential_barycenter = torch.sum(aux_pos * energies, axis=0)
        hf_potential_term = torch.zeros(3)
        for alpha in range(3):
            tmp = (
                torch.autograd.grad(
                    potential_barycenter[alpha],
                    pos,
                    retain_graph=True,
                )[0]
                .detach()
                .squeeze()
            )
            hf_potential_term[alpha] = torch.sum(tmp * velocities)

        hf_potential_term = hf_potential_term.cpu().numpy()

        energy = energies.sum()

        gradient = (
            torch.autograd.grad(energy, pos, retain_graph=False)[0]
            .detach()
            .squeeze()
        )

        inner = (gradient * velocities).sum(axis=1)
        hf_force_term = (
            (pos * inner.unsqueeze(1)).sum(axis=0).detach().cpu().numpy()
        )

        heat_flux = (hf_potential_term - hf_force_term) / volume

        self.results.update({
            "heat_flux": heat_flux,
            "heat_flux_force_term": hf_force_term,
            "heat_flux_potential_term": hf_potential_term,
        })

        return self.results
