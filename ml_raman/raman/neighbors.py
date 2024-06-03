from mlcalcdriver.interfaces import posinp_to_ase_atoms, AtomsToPatches
from schnetpack.environment import AseEnvironmentProvider


def precalculate_patches_and_environments(posinp, grid, cutoff, n_interactions):
    atoms = posinp_to_ase_atoms(posinp)

    at_to_patches = AtomsToPatches(
        cutoff=cutoff, n_interaction=n_interactions, grid=grid
    )
    (subcells, subcells_main_idx, original_cell_idx, complete_subcell_copy_idx) = (
        at_to_patches.split_atoms(atoms)
    )

    env_provider = AseEnvironmentProvider(cutoff=cutoff)

    environments = []
    for subcell in subcells:
        environments.append(env_provider.get_environment(subcell))
    return environments, (
        subcells,
        subcells_main_idx,
        original_cell_idx,
        complete_subcell_copy_idx,
    )
