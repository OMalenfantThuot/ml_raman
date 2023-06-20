from ase.phonons import Phonons
from ase.io import read
from ase import Atoms
from ml_raman.dos import Dos
from schnetpack.utils import load_model
from schnetpack.environment import AseEnvironmentProvider
from schnetpack.interfaces import SpkCalculator
import h5py
import torch


def get_dos(
    model,
    atoms,
    device="cpu",
    supercell=(6, 6, 6),
    qpoints=[30, 30, 30],
    npts=1000,
    width=0.004,
):
    if isinstance(atoms, str):
        atoms = read(atoms)
    elif isinstance(atoms, Atoms):
        pass
    else:
        raise ValueError("The posinp variable is not recognized.")

    if isinstance(model, str):
        model = load_model(model, map_location=device)
    elif isinstance(model, torch.nn.Module):
        pass
    else:
        raise ValueError("The model variable is not recognized.")

    assert len(supercell) == 3, "Supercell should be a length 3 object."
    assert len(qpoints) == 3, "Qpoints should be a length 3 object."
    supercell = tuple(supercell)

    cutoff = float(
        model.state_dict()["representation.interactions.0.cutoff_network.cutoff"]
    )
    calculator = SpkCalculator(
        model,
        device=device,
        energy="energy",
        forces="forces",
        environment_provider=AseEnvironmentProvider(cutoff),
    )
    ph = Phonons(atoms, calculator, supercell=supercell, delta=0.02)
    ph.run()
    ph.read(acoustic=True)
    dos = ph.get_dos(kpts=qpoints).sample_grid(npts=npts, width=width)
    ph.clean()
    return Dos(dos.get_energies() * 8065.6, dos.get_weights())
