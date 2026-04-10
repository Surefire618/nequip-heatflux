import numpy as np
import torch

from ase import units
from ase.calculators.calculator import Calculator, all_changes

from nequip.ase import NequIPCalculator
from nequip.data import AtomicData, AtomicDataDict

from .unfolder import Unfolder

# nequip wraps the energy model in these to add autograd-based outputs.
# Their forward() calls torch.autograd.grad internally without retain_graph,
# which frees the graph we need for the heat-flux autograd. We strip them
# and call the inner energy module directly for the unfolded pass.
_GRAD_WRAPPER_NAMES = (
    "StressOutput",
    "StressForceOutput",
    "GradientOutput",
    "ForceOutput",
    "PartialForceOutput",
    "StrainStressOutput",
)


def _strip_grad_wrappers(model):
    cur = model
    while type(cur).__name__ in _GRAD_WRAPPER_NAMES and hasattr(cur, "func"):
        cur = cur.func
    return cur


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

        self.energy_model = _strip_grad_wrappers(self.model)
        if self.energy_model is self.model:
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
        data = self.energy_model(data)
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
