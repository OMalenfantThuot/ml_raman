from ase import Atoms
from ase.phonons import Phonons
from ase.spectrum.dosdata import GridDOSData
from mlcalcdriver.calculators.ase_calculators import AseSpkCalculator

eV_to_cm = 8065.73


def calculate_phonons(atoms: Atoms, model, supercell, device="cpu"):
    if isinstance(atoms, str):
        atoms = read(atoms)
    elif isinstance(atoms, Atoms):
        pass
    else:
        raise ValueError("The posinp variable is not recognized.")

    assert len(supercell) == 3, "Supercell should be a length 3 object."
    supercell = tuple(supercell)
    print(supercell)

    calculator = AseSpkCalculator(model, device=device)

    phonons = Phonons(atoms, calculator, supercell=supercell, delta=0.01)
    phonons.run()
    phonons.read(acoustic=True)
    phonons.clean()
    return phonons


def phonons_band_structure(
    atoms: Atoms, phonons: Phonons, path: str, npoints: int = 300
):
    path = atoms.cell.bandpath(path, npoints=npoints)
    band_structure = phonons.get_band_structure(path)
    band_structure._energies *= eV_to_cm
    return band_structure


def phonons_dos(phonons: Phonons, qpoints: list, npts: int, width: float):
    assert len(qpoints) == 3, "Qpoints should be a length 3 object."

    dos = phonons.get_dos(kpts=qpoints).sample_grid(npts=npts, width=width)
    dos = GridDOSData(energies=dos.get_energies() * eV_to_cm, weights=dos.get_weights())
    return dos
