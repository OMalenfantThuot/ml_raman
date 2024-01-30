#!/usr/bin/env python
import logging
from mlcalcdriver import Posinp
from mlcalcdriver.globals import ATOMS_MASS
from ase.io import read
from ase.build.supercells import make_supercell
import numpy as np
import random
import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "unitcell", help="Path to position file containing the equilibrium positions of unit cell."
    )
    parser.add_argument(
        "supercell",
        help="Size of the supercell.",
        nargs=3,
        type=int,
    )
    parser.add_argument(
        "dopant_concentration",
        help="Concentration of added dopant.",
        type = float
    )
    parser.add_argument(
        "orig_name",
        help="Names of original element.",
        type=str
    )
    parser.add_argument(
        "dopant_name",
        help="Name of added dopant.",
        type=str
    )
    parser.add_argument(
        "output_name", help="Name of the written file.", type=str
    )

    return parser

def main(args):
    #Make supercell
    if 0 <=args.dopant_concentration<= 1:
        Mx, My, Mz = args.supercell
        M = [[Mx, 0, 0], [0, My, 0], [0, 0, Mz]]
        cif = read(args.unitcell)
        sc=make_supercell(cif, M)
        posinp=Posinp.from_ase(sc)
        n_atoms = len(sc.positions)
        n_ind = np.floor(args.dopant_concentration*n_atoms).astype(int) #number of dopant atoms
        array = np.arange(0,n_atoms) #array of atoms' indices
        dopant = random.sample(array.tolist(),n_ind)
        not_dopant = [i for i in array if i not in dopant]
          
        for idx in dopant:
            posinp.atoms[idx].type = args.dopant_name
            #posinp[idx].atom.mass =  ATOMS_MASS[posinp[idx].atom.type]
        posinp.write(args.output_name)

    else:
        raise ValueError("The dopant concentration value is not between 0 and 1.")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
