import torch
from torch.autograd import grad
import numpy as np

from ase import Atom, units
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes


from nequip.ase import NequIPCalculator
from nequip.data import AtomicData, AtomicDataDict

class HardyCalculator(NequIPCalculator):
    implemented_properties = [
        "energy",
        "forces",
        "stress",
        "energies",
    ]

    def __init__(
        self,
        barycenter:bool=False,
        *args,
        **kwargs,
    ):
        NequIPCalculator.__init__(self, *args, **kwargs)
        self.barycenter = barycenter


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

        virials_reference, barycenter_reference = get_reference(atoms)
        self.barycenter_reference = barycenter_reference

        # predict + extract data
        self.results = {}

        ref = virials_reference
        pos = data[AtomicDataDict.POSITIONS_KEY]

        velocities = torch.tensor(atoms.get_velocities() * units.fs * 1000).to(self.device)

        pos.requires_grad_(True)
        data = self.model(data)

        energies = data[AtomicDataDict.PER_ATOM_ENERGY_KEY]

        virials = compute_hardy_virials(
            ref=ref.to(self.device),
            energies=energies,
            r=pos,
            volume=atoms.get_volume(),
        )

        self.results["virials"] = virials.detach().cpu().numpy()
        self.results["stresses"] = - self.results["virials"]
        self.results["stress"] = self.results["stresses"].sum(axis=0)
        # self.results["stress"] = data[AtomicDataDict.STRESS_KEY].detach().squeeze().cpu().numpy()

        forces = data[AtomicDataDict.FORCE_KEY].detach().squeeze()
        self.results["forces"] = forces.cpu().numpy()

        if self.barycenter:
            from .barycenter import compute_barycenter

            energies = data[AtomicDataDict.PER_ATOM_ENERGY_KEY]
            positions = data[AtomicDataDict.POSITIONS_KEY]
            velocities = torch.tensor(atoms.get_velocities() * units.fs * 1000).to(
                self.device
            )
            volume = atoms.get_volume()

            if self.barycenter_reference is not None:
                r0_j, r0_j_pot, r0_j_kin = compute_barycenter(
                    reference=self.barycenter_reference.to(self.device),
                    energies=energies,
                    positions=positions,
                    forces=forces,
                    velocities=velocities,
                    volume=volume,
                )

                self.results["barycenter_r0_heat_flux"] = r0_j
                self.results["barycenter_r0_heat_flux_pot"] = r0_j_pot
                self.results["barycenter_r0_heat_flux_kin"] = r0_j_kin

            rt_j, rt_j_pot, rt_j_kin = compute_barycenter(
                reference=positions.detach().squeeze(),
                energies=energies,
                positions=positions,
                forces=forces,
                velocities=velocities,
                volume=volume,
            )

            self.results["barycenter_rt_heat_flux"] = rt_j
            self.results["barycenter_rt_heat_flux_pot"] = rt_j_pot
            self.results["barycenter_rt_heat_flux_kin"] = rt_j_kin

        energies = data[AtomicDataDict.PER_ATOM_ENERGY_KEY].detach().squeeze().cpu().numpy()
        self.results["energies"] = energies.reshape(len(atoms))
        energy = data[AtomicDataDict.TOTAL_ENERGY_KEY].detach().squeeze().cpu().numpy()
        self.results["energy"] = energy


        vs = atoms.get_velocities() * units.fs
        fluxes = np.squeeze(self.results["virials"] @ vs[:, :, None])
        heat_flux = fluxes.sum(axis=0)
        self.results["heat_flux"] = heat_flux

        return self.results


def compute_virials(du_drij, r_ij, hatr_ij, volume, vecr_ij=None):
    """Compute virials."""

    # derivative wrt displacement vectors
    du_dvecrij = du_drij.unsqueeze(-1) * hatr_ij

    # displacment vectors
    if vecr_ij is None:
        vecr_ij = r_ij.unsqueeze(-1) * hatr_ij

    # outer product across cartesian dimensions
    # (order is arbitrary; it's symmetric anyway)
    # sign also doesn't matter, but the derivation is slightly more
    # consistent with the -1!
    virials = -1.0 * vecr_ij.unsqueeze(-1) * du_dvecrij.unsqueeze(-2)

    # sum over j
    virials = torch.sum(virials, axis=1)

    return virials / volume


def compute_hardy_virials(ref, energies, r, volume):
    """Compute virials with the Hardy formula

    In particular, we compute:

        sum_i r_ji (d U_i)/(d r_j); i.e. a [j, 3] tensor.

    Args:
        ref: [i, j, a] Reference positions (should be pairwise mic distance vectors)
        energies: [1, i, 1] Atomic energies
        r: [1, i, a] Positions (must be the tensor used to compute ui, we need gradients)
        volume: scalar, Volume of unit cell (vibes expects intensive virials)

    Note that this is not vectorised across a batch; the calculator
    does one structure at a time anyways.

    """

    virials = torch.zeros(energies.shape[0], 3, 3, device=energies.device)
    for i in range(energies.shape[0]):
        rji = -1.0 * ref[i, :, :]  # => [j, a] // sign due to convention
        dui_drj = torch.autograd.grad(
            energies[i],
            r,
            retain_graph=True
        )[0].squeeze(0) # => [j, b]

        # outer product;
        # see https://discuss.pytorch.org/t/easy-way-to-compute-outer-products/720/5
        virials += rji.unsqueeze(2) * dui_drj.unsqueeze(1)  # => [j, a, b]

    return virials / volume


def get_reference(atoms):
    from ase import Atoms

    if not isinstance(atoms, Atoms):
        comms.talk(f"loading reference positions for virials from {atoms}", full=True)
        from ase.io import read

        atoms = read(atoms, format="aims")

    return torch.tensor(atoms.get_all_distances(vector=True, mic=True)), torch.tensor(
        atoms.get_positions()
    )
