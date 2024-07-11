from mlcalcdriver import Posinp
from mlcalcdriver.globals import ATOMS_MASS
from ase.io import read
from ase.build.supercells import make_supercell
from mlcalcdriver.calculators import PatchSPCalculator
from mlcalcdriver.workflows import Geopt
from mlcalcdriver.interfaces import posinp_to_ase_atoms
import numpy as np
import random
import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "primitive_cell", help="path to primitive cell .cif file.", type=str
    )
    parser.add_argument(
        "supercell",
        help="""
        Size of the supercell of the form int int. Third argument is taken
        to be 1 since we are working with a 2D system.
        """,
        nargs=2,
        type=int,
    )
    parser.add_argument("output_name", help="name of written file.", type=str)
    parser.add_argument(
        "--step", help="step size for relaxation.", type=float, default=0.01
    )
    parser.add_argument(
        "--n_vacancies", help="Number of vacancies. Default is 0.", type=int, default=0
    )
    parser.add_argument(
        "--vacancies_file",
        help="Path to the file containing the vacancies index.",
        default=None,
    )
    parser.add_argument(
        "--relax",
        action=argparse.BooleanOptionalAction,
        help="Option to relax structure",
        default=True,
    )
    parser.add_argument(
        "--precalculate_neighbors",
        action=argparse.BooleanOptionalAction,
        help="Option to precalculate neighbors for the relaxation.",
        default=True,
    )
    parser.add_argument(
        "--model",
        help="Path to model used to relax structure. Only used if --relax True.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        help="Can be either cpu to use cpu or cuda to use gpu.",
        default="cpu",
        type=str,
    )
    parser.add_argument(
        "--output_format",
        choices=["mlc", "xyz", "cif"],
        default="mlc",
        help="Format of the output file.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Max iterations for geometry optimization",
    )
    parser.add_argument("--restart_file", default=None)
    return parser


def main(args):
    Mx, My = args.supercell
    if args.restart_file:
        posinp = Posinp.from_file(args.restart_file)
    else:
        cif = read(args.primitive_cell)
        n_vac = args.n_vacancies
        M = [[Mx, 0, 0], [0, My, 0], [0, 0, 1]]
        sc = make_supercell(cif, M)
        posinp = Posinp.from_ase(sc)

        if args.vacancies_file is not None:
            indices_to_remove = np.load(args.vacancies_file)
        else:
            indices_to_remove = random.sample(range(len(posinp.atoms)), n_vac)
            indices_to_remove.sort()
        indices_to_remove = np.flip(indices_to_remove)

        for index in indices_to_remove:
            _ = posinp.atoms.pop(index)

        unrelaxed_output = args.output_name + "_unrelaxed." + args.output_format
        if args.output_format == "mlc":
            posinp.write(unrelaxed_output)
        elif args.output_format in ["xyz", "cif"]:
            ase_unrelaxed = posinp_to_ase_atoms(posinp)
            ase_unrelaxed.write(unrelaxed_output)

    if args.relax:
        # Hard coded for graphene and hBN
        posinp = posinp.translate([0.62, 0.36, posinp.cell[2, 2] / 2])

        from ml_raman.raman.neighbors import precalculate_patches_and_environments
        from utils.models import get_graphene_patches_grid, get_schnet_hyperparams

        n_neurons, n_interactions, cutoff = get_schnet_hyperparams(args.model)
        patches_grid = get_graphene_patches_grid(
            "forces", n_interactions, cutoff, n_neurons, Mx, My
        )
        if args.precalculate_neighbors:
            atomic_environments, _ = precalculate_patches_and_environments(
                posinp, patches_grid, cutoff, n_interactions
            )
        else:
            atomic_environments = None

        calc = PatchSPCalculator(
            args.model,
            device=args.device,
            subgrid=patches_grid,
            atomic_environments=atomic_environments,
        )
        g = Geopt(
            posinp, calc, step_size=args.step, max_iter=args.max_iter, forcemax=0.015
        )
        g.run(verbose=2)

        # Undo first translation
        final_posinp = g.best_posinp.translate(
            [-0.62, -0.36, -1.0 * posinp.cell[2, 2] / 2]
        )

        relaxed_output = args.output_name + "_relaxed." + args.output_format
        if args.output_format == "mlc":
            final_posinp.write(relaxed_output)
        elif args.output_format in ["xyz", "cif"]:
            ase_relaxed = posinp_to_ase_atoms(final_posinp)
            ase_relaxed.write(relaxed_output)
    else:
        pass


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
