# HIGHLY experimental
import numpy as np
import torch

from ase import units, Atoms
from ase.calculators.calculator import Calculator, all_changes

from nequip.ase import NequIPCalculator
from nequip.data import AtomicData, AtomicDataDict

from .unfolder_calculator import _find_grad_wrapper_parent, _spliced_grad_wrapper


class FoldedHeatFluxCalculator(NequIPCalculator):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        NequIPCalculator.__init__(self, *args, **kwargs)

        self._grad_wrapper_location = _find_grad_wrapper_parent(self.model)
        if self._grad_wrapper_location is None:
            print(
                "FoldedHeatFluxCalculator: no GradientOutput/StressOutput "
                "wrapper found on the loaded model; the heat-flux autograd "
                "may fail if the model frees its graph internally."
            )

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        n = len(atoms)

        # first pass: wrapped model for energy/forces/stress
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]
        data = self.transform(data)
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        data = self.model(data)

        self.results = {}
        self.results["forces"] = data[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        if AtomicDataDict.STRESS_KEY in data:
            self.results["stress"] = data[AtomicDataDict.STRESS_KEY].detach().squeeze().cpu().numpy()
        self.results["energies"] = data[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().cpu().numpy()
        self.results["energy"] = self.results["energies"].sum()

        del data

        # second pass: bare energy model so the autograd graph survives
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]
        data = self.transform(data)
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        pos = data[AtomicDataDict.POSITIONS_KEY]
        aux_pos = pos.detach()

        velocities = torch.tensor(atoms.get_velocities() * units.fs).to(self.device)

        pos.requires_grad_(True)
        with _spliced_grad_wrapper(self._grad_wrapper_location):
            data = self.model(data)

        energies = data[AtomicDataDict.PER_ATOM_ENERGY_KEY]

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
        hf_force_term = (pos * inner.unsqueeze(1)).sum(axis=0).detach().cpu().numpy()

        heat_flux = (hf_potential_term - hf_force_term) / atoms.get_volume()

        self.results.update({
            "heat_flux": heat_flux,
            "heat_flux_force_term": hf_force_term,
            "heat_flux_potential_term": hf_potential_term,
        })

        return self.results

