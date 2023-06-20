#! /usr/bin/env python

import argparse
from ml_raman.dos import get_dos


def main(args):
    dos = get_dos(
        args.model,
        args.posinp,
        device=args.device,
        supercell=args.supercell,
        qpoints=args.qpoints,
        npts=args.npts,
        width=args.width,
    )
    dos.write(args.name)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model", help="Path to the model used to generate the phonon dos."
    )
    parser.add_argument(
        "posinp", help="Path to position file containing the equilibrium positions."
    )
    parser.add_argument("--device", default="cpu", help="Either 'cuda' or 'cpu'.")
    parser.add_argument(
        "--supercell",
        help="Size of the supercell.",
        nargs=3,
        default=[6, 6, 6],
        type=int,
    )
    parser.add_argument(
        "--qpoints",
        help="Qpoints grid for the dos estimation.",
        nargs=3,
        type=int,
        default=[30, 30, 30],
    )
    parser.add_argument(
        "--npts", help="Resolution in energy for the dos.", default=1000, type=int
    )
    parser.add_argument(
        "--width", help="Gaussian smearing to build the dos.", default=0.004, type=float
    )
    parser.add_argument("name", help="Name of the dos written on file.", type=str)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    args.supercell = tuple(args.supercell)
    dos = main(args)
