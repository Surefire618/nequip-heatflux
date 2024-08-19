# HIGHLY experimental
import numpy as np
import torch

from ase import units, Atoms
from ase.calculators.calculator import Calculator, all_changes

from nequip.ase import NequIPCalculator
from nequip.data import AtomicData, AtomicDataDict

class FoldedHeatFluxCalculator(NequIPCalculator):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        NequIPCalculator.__init__(self, *args, **kwargs)

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        data = AtomicData.from_ase(atoms=atoms, r_max=self.r_max)
        for k in AtomicDataDict.ALL_ENERGY_KEYS:
            if k in data:
                del data[k]
        data = self.transform(data)
        data = data.to(self.device)
        data = AtomicData.to_AtomicDataDict(data)

        # predict + extract data
        self.results = {}

        pos = data[AtomicDataDict.POSITIONS_KEY]
        aux_pos = pos.detach()

        n = len(atoms)

        velocities = torch.tensor(atoms.get_velocities() * units.fs * 1000).to(self.device)

        pos.requires_grad_(True)
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
            torch.autograd.grad(energy, pos, retain_graph=True)[0]
            .detach()
            .squeeze()
        )

        inner = (gradient * velocities).sum(axis=1)
        hf_force_term = (pos * inner.unsqueeze(1)).sum(axis=0).detach().cpu().numpy()

        heat_flux = (hf_potential_term - hf_force_term) / atoms.get_volume()

        self.results = {
            "heat_flux": heat_flux,
            "heat_flux_force_term": hf_force_term,
            "heat_flux_potential_term": hf_potential_term,
        }

        self.results["forces"] = data[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        self.results["stress"] = data[AtomicDataDict.STRESS_KEY].detach().squeeze().cpu().numpy()
        self.results["energies"] = energies.detach().cpu().numpy()
        self.results["energy"] = self.results["energies"].sum()

        return self.results

