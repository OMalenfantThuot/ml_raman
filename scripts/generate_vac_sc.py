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
    parser.add_argument(
        "n_vacancies", help="Number of vacancies. Default is 0.", type=int, default=0
    )
    parser.add_argument(
        "output_name", help="name of written file.", type = str
    )    
    parser.add_argument("--relax", help="option to relax structure")
    parser.add_argument(
        "model",
        help="path to model used to relax structure. Only used if --relax true.",
        type=str,
    )

    return parser


def main(args):
    cif = read(args.primitive_cell)
    Mx, My = args.supercell
    n_vac = args.n_vacancies
    calc = SchnetPackCalculator(args.model)
    M = [[Mx, 0, 0], [0, My, 0], [0, 0, 1]]
    sc = make_supercell(cif, M)
    posinp = Posinp.from_ase(sc)
    indices_to_remove = random.sample(range(len(posinp.atoms)), n_vac)
    indices_to_remove.sort(reverse=True)
    for index in indices_to_remove:
        removed_element = posinp.atoms.pop(index)
    g = Geopt(posinp, calc)
    g.run(verbose=2)
    posinp_ase_unrelaxed = posinp_to_ase_atoms(posinp)
    filename_unrelaxed = (
        args.output_name
        if args.output_name.endswith(".xyz")
        else args.output_name + ".xyz"
    )
    filename_relaxed = (
         args.output_name[:-4]+"_relaxed.xyz"
        if args.output_name.endswith(".xyz")
        else args.output_name + "_relaxed.xyz"
    )

    posinp_ase_unrelaxed.write(filename_unrelaxed)
    g_ase = posinp_to_ase_atoms(g.final_posinp)
    g_ase.write(filename_relaxed)
    filename_vacidx = (
        args.output_name[:-4]+"_vacidx.dat"
        if args.output_name.endswith(".xyz")
        else args.output_name +"_vacidx.dat"
    )
    np.savetxt(filename_vacidx,indices_to_remove)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
