from mlcalcdriver import Posinp
from mlcalcdriver.interfaces import posinp_to_ase_atoms
from ase import io
from ase.io import read, write
from ase.build.supercells import make_supercell
import numpy as np
import random
from math import isclose
import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "unitcell",
        help="Path to position file containing the equilibrium positions of unit cell.",
    )
    parser.add_argument(
        "supercell",
        help="Size of the supercell.",
        nargs=3,
        type=int,
    )
    parser.add_argument("n_sw", help="Number of stone-wales defects.", type=float)
    parser.add_argument("output_name", help="Name of the written file.", type=str)

    return parser


def main(args):
    # Make supercell
    Mx, My, Mz = args.supercell
    M = [[Mx, 0, 0], [0, My, 0], [0, 0, Mz]]
    cif = read(args.unitcell)
    sc = make_supercell(cif, M)
    n_atoms = len(sc.positions)
    n_sw = args.n_sw
    # print(sc.positions)
    # Store n_sw unique pairs of atom positions with each pair having the same x coordinate.
    pairs = []
    # Array of atoms' indices.
    atoms_idx = np.arange(0, n_atoms)
    # Make sure we don't choose atoms on edge, as they don't come in a pair with same x coordinate.
    atoms_idx_edge1 = np.arange(0, n_atoms, 2 * My)
    atoms_idx_edge2 = np.arange(2 * My - 1, n_atoms, 2 * My)
    # print("atoms_idx_edge1= ", atoms_idx_edge1)
    # print("atoms_idx_edge2= ", atoms_idx_edge2)
    atoms_idx_edge = sorted(np.concatenate((atoms_idx_edge1, atoms_idx_edge2)))
    atoms_idx_no_edge = [i for i in atoms_idx if i not in atoms_idx_edge]
    # print("atoms_idx_no_edge = ", atoms_idx_no_edge)
    while len(pairs) < n_sw:
        # Choose a random index to start the pair
        idx0 = random.randint(0, len(atoms_idx_no_edge) - 2)
        # Ensure that the index is even
        if idx0 % 2 == 1:
            idx0 -= 1
        # Get the two consecutive points
        pair = (atoms_idx_no_edge[idx0], atoms_idx_no_edge[idx0 + 1])
        # Check if the pair is not already in the list
        if pair not in pairs:
            # Add the pair to the list
            pairs.append(pair)
    print(len(pairs))
    for idx in pairs:
        x1 = sc.positions[idx[0]][0]
        y1 = sc.positions[idx[0]][1]
        x2 = sc.positions[idx[1]][0]
        y2 = sc.positions[idx[1]][1]
        # print("x1= ", x1)
        # print("y1= ", y1)
        # print("x2= ", x2)
        # print("y2= ", y2)
        # Make sure both atoms have same y coordinate
        assert isclose(x1, x2, abs_tol=1e-4)
        # Rotated coordinates
        x1_pr = x1 - (y2 - y1) / 2
        y1_pr = (y1 + y2) / 2
        x2_pr = x2 + (y2 - y1) / 2
        y2_pr = (y1 + y2) / 2
        # Rotate atoms in sc
        sc.positions[idx[0]][0] = x1_pr
        sc.positions[idx[0]][1] = y1_pr
        sc.positions[idx[1]][0] = x2_pr
        sc.positions[idx[1]][1] = y2_pr
    # print(sc.positions)
    write(args.output_name, sc)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
