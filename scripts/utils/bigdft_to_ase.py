import argparse
import ase
import numpy as np
from ase.units import Bohr


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("infile", help="BigDFT position file, in ascii format")
    return parser


def main(args):
    with open(args.infile, "r") as f:
        lines = f.readlines()

    cell_lines = lines[1:3]
    cell_values = [float(v) for l in lines[1:3] for v in l.split()]
    cell = ase.geometry.Cell(
        [[cell_values[0], 0, 0], [0, cell_values[5], 0], [0, 0, cell_values[2]]]
    )

    positions_lines = [l for l in lines if not l.startswith("#")][2:]
    positions = [[float(v) for v in l.split()[:-1]] for l in positions_lines]
    positions = np.array(positions)
    positions[:, [1, 2]] = positions[:, [2, 1]]

    keywords_lines = [l.rstrip("\n") for l in lines if l.startswith("#keyword")]
    if "#keyword: atomicd0" in keywords_lines:
        positions *= Bohr

    elements = ""
    for l in positions_lines:
        elements += l.split()[-1]

    atoms = ase.Atoms(elements, positions=positions, cell=cell, pbc=True)
    atoms.write("out.xyz")


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
