# /usr/bin/env python

from mlcalcdriver import Posinp
from mlcalcdriver.calculators import PatchSPCalculator
from ml_raman.raman.lanczos import lanczos_raman_projections
from ml_raman.raman.neighbors import precalculate_patches_and_environments
from utils.models import get_graphene_patches_grid, get_schnet_hyperparams
from schnetpack.utils import load_model
import argparse
import h5py
import numpy as np
import time


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("posinp", help="Path to the position file.")
    parser.add_argument("model", help="Path to the model.")
    parser.add_argument(
        "material", choices=["graphene", "BN"], help="Material for projections."
    )
    parser.add_argument(
        "--grid",
        default=None,
        type=int,
        help="Size of the grid. If None, will be chosen by the code.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Either 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--k", default=None, type=int, help="Number of eigenvalues to compute."
    )
    parser.add_argument(
        "--eigen_proportion",
        default=None,
        type=float,
        help="Proportion of eigenvalues to compute, in percent.",
    )
    parser.add_argument(
        "--results_savepath",
        default="phonons.h5",
        help="Name of the output file for the phonon results.",
    )
    return parser


def main(args):
    try:
        posinp = Posinp.from_file(args.posinp)
    except Exception as e:
        print(f"ML_Calc_Driver read exception: {str(e)}")
        try:
            posinp = Posinp.read(args.posinp)
        except Exception as e:
            print(f"ASE read exception: {str(e)}")
            exit()

    # Center the positions so that small displacements don't cross the boundary
    if args.material == "graphene":
        assert (posinp[0].position[0:2] == np.array([0, 0])).all()
        posinp = posinp.translate([0.61, 0.35, posinp.cell[2, 2] / 2])
    elif args.material == "BN":
        posinp = posinp.translate([0.62, 0.36, posinp.cell[2, 2] / 2])
    else:
        raise NotImplementedError()

    if args.k is None and args.eigen_proportion is None:
        raise RuntimeError(
            "Choose one of the arguments to set the number of eigenvalues to compute."
        )
    elif args.k is not None and args.eigen_proportion is not None:
        raise RuntimeError(
            "Choose only one of the arguments to set the number of eigenvalues to compute."
        )
    elif args.eigen_proportion is not None:
        k = int(args.eigen_proportion / 100 * 3 * len(posinp))
    else:
        k = int(args.k)

    n_neurons, n_interactions, cutoff = get_schnet_hyperparams(args.model)

    if args.grid is None:
        size = int(np.sqrt(len(posinp) / 2))
        patches_grid = get_graphene_patches_grid(
            "forces", n_interactions, cutoff, n_neurons, size, size
        )
    else:
        patches_grid = np.array([args.grid, args.grid, 1])
    print("Patches grid: ", patches_grid)
    atomic_environments, patches = precalculate_patches_and_environments(
        posinp, patches_grid, cutoff, n_interactions
    )

    calculator = PatchSPCalculator(
        args.model,
        device=args.device,
        subgrid=patches_grid,
        atomic_environments=atomic_environments,
        patches=None,
    )

    eigs, eigvs = lanczos_raman_projections(
        posinp, calculator, material=args.material, k=k, patches=patches
    )

    savename = (
        args.results_savepath
        if args.results_savepath.endswith(".h5")
        else args.results_savepath + ".h5"
    )
    with h5py.File(savename, "w") as f:
        f.create_dataset("modes", data=eigvs)
        f.create_dataset("energies", data=eigs)


if __name__ == "__main__":
    t1 = time.time()
    parser = create_parser()
    args = parser.parse_args()
    main(args)
    print("Timing total: ", time.time() - t1)
