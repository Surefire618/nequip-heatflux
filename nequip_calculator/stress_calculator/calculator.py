import numpy as np
import torch

from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from ase import units

from nequip.ase import NequIPCalculator
from nequip.data import AtomicData, AtomicDataDict

class StressCalculator(NequIPCalculator, torch.nn.Module):
    r"""Compute atomic stress from dui/dstrain.

    Args:
        func: the energy model to wrap
        do_forces: whether to compute forces as well
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        NequIPCalculator.__init__(self, *args, **kwargs)
        self.register_buffer("_empty", torch.Tensor())

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

        cell = data[AtomicDataDict.CELL_KEY]
        # Make the cell per-batch
        data[AtomicDataDict.CELL_KEY] = cell

        # strain-stress and atomic stress
        # Add the strain
        strain = torch.zeros(
            (3, 3),
            dtype=pos.dtype,
            device=pos.device,
        )
        strain.requires_grad_(True)
        data["_strain"] = strain
        did_pos_req_grad: bool = pos.requires_grad
        pos.requires_grad_(True)
        cell.requires_grad_(True)

        def wrapper(strain: torch.Tensor) -> torch.Tensor:
            nonlocal data
            # [natom, 3] @ [3, 3] -> [natom, 3]
            data[AtomicDataDict.POSITIONS_KEY] = torch.addmm(
                pos, pos, strain
            )
            # assert torch.equal(pos, data[AtomicDataDict.POSITIONS_KEY])
            data[AtomicDataDict.CELL_KEY] = cell + cell @ strain

            # Call model and get gradients
            data = self.model(data)
            return data[AtomicDataDict.PER_ATOM_ENERGY_KEY].squeeze(-1)

        # get atomic stress (very inefficient implementation)
        # d Ei / d strain [natom] / [3,3] -> [natom, 3, 3]
        jac = torch.autograd.functional.jacobian(
            func=wrapper,
            inputs=data["_strain"],
            create_graph=self.training,  # needed to allow gradients of this output during training
            # vectorize=self.vectorize,
            # strategy="forward-mode",
        )
        # jac = jacfwd(wrapper)(data["_strain"]) # this doesn't work, and I don't understand why!
        atomic_virial = jac

        # Store virial
        virial = atomic_virial.sum(axis=0)
        virial = (virial + virial.transpose(-1, -2)) / 2 # symmetric

        stress = virial / atoms.get_volume()
        data[AtomicDataDict.CELL_KEY] = cell

        virial = torch.neg(virial)
        atomic_stress = torch.neg(atomic_virial)
        data[AtomicDataDict.STRESS_KEY] = stress
        data[AtomicDataDict.VIRIAL_KEY] = virial
        data[AtomicDataDict.PER_ATOM_STRESS_KEY] = atomic_stress / atoms.get_volume()


        # gather all results
        self.results = {}
        self.results["energy"] = self.energy_units_to_eV * data[AtomicDataDict.TOTAL_ENERGY_KEY].detach().squeeze().cpu().numpy()
        self.results["free_energy"] = self.results["energy"]
        self.results["energies"] = self.energy_units_to_eV * data[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().squeeze().cpu().numpy()
        self.results["forces"] = self.energy_units_to_eV / self.length_units_to_A * data[AtomicDataDict.FORCE_KEY].detach().cpu().numpy()
        stress = data[AtomicDataDict.STRESS_KEY].detach().cpu().numpy()
        stress = stress.reshape(3, 3) * (self.energy_units_to_eV / self.length_units_to_A**3)
        stress_voigt = full_3x3_to_voigt_6_stress(stress)
        self.results["stress"] = stress_voigt

        stresses = data[AtomicDataDict.PER_ATOM_STRESS_KEY].detach().cpu().numpy()
        stresses *= (self.energy_units_to_eV / self.length_units_to_A**3)
        # stresses_voigt = np.asarray([full_3x3_to_voigt_6_stress(stress) for stress in stresses])
        self.results["stresses"] = stresses

        vs = atoms.get_velocities() * units.fs * 1000
        fluxes = np.squeeze(self.results["stresses"] @ vs[:, :, None])
        heat_flux = fluxes.sum(axis=0)
        self.results["heat_flux"] = heat_flux

        return self.results
