from mlcalcdriver import Posinp
from mlcalcdriver.globals import ATOMS_MASS
from ase.io import read
from ase.build.supercells import make_supercell
from mlcalcdriver.calculators import SchnetPackCalculator
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
        help="size of the supercell of the form int int. Third argument is taken to be 1 since we are working with a 2D system.",
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
        "--vacancies_indices",
        help="option to choose specific indices for vacancies.",
        type=int,
        nargs="*",
    )
    parser.add_argument(
        "--relax",
        action=argparse.BooleanOptionalAction,
        help="option to relax structure",
        default=True,
    )
    parser.add_argument(
        "--model",
        help="path to model used to relax structure. Only used if --relax True.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        help="Can be either cpu to use cpu or cuda to use gpu.",
        default="cpu",
        type=str,
    )
    return parser


def main(args):
    cif = read(args.primitive_cell)
    Mx, My = args.supercell
    n_vac = args.n_vacancies
    M = [[Mx, 0, 0], [0, My, 0], [0, 0, 1]]
    sc = make_supercell(cif, M)
    posinp = Posinp.from_ase(sc)
    if args.vacancies_indices is not None:
        indices_to_remove = args.vacancies_indices
    else:
        indices_to_remove = random.sample(range(len(posinp.atoms)), n_vac)
    indices_to_remove.sort(reverse=True)
    for index in indices_to_remove:
        removed_element = posinp.atoms.pop(index)
    if args.relax == True:
        calc = SchnetPackCalculator(args.model, device=args.device)
        g = Geopt(posinp, calc, step_size=args.step, max_iter=1000)
        g.run(verbose=2)
        filename_relaxed = (
            args.output_name[:-4] + "_relaxed.xyz"
            if args.output_name.endswith(".xyz")
            else args.output_name + "_relaxed.xyz"
        )
        g_ase = posinp_to_ase_atoms(g.final_posinp)
        g_ase.write(filename_relaxed)
    else:
        pass

    posinp_ase_unrelaxed = posinp_to_ase_atoms(posinp)
    filename_unrelaxed = (
        args.output_name
        if args.output_name.endswith(".xyz")
        else args.output_name + ".xyz"
    )

    posinp_ase_unrelaxed.write(filename_unrelaxed)
    filename_vacidx = (
        args.output_name[:-4] + "_vacidx.dat"
        if args.output_name.endswith(".xyz")
        else args.output_name + "_vacidx.dat"
    )
    np.savetxt(filename_vacidx, indices_to_remove)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
