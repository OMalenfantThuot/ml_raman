#! /usr/bin/env python

from ase.io import read
from ml_raman.phonons.ase_phonons import (
    calculate_phonons,
    phonons_band_structure,
    phonons_dos,
)
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pickle


def main(args):
    atoms = read(args.posinp)
    phonons = calculate_phonons(
        atoms=atoms, model=args.model, supercell=args.supercell, device=args.device
    )
    if args.type in ["band", "full"]:
        band_structure = phonons_band_structure(
            atoms=atoms, phonons=phonons, path=args.path
        )
    else:
        band_structure = None

    if args.type in ["dos", "full"]:
        dos = phonons_dos(
            phonons=phonons, qpoints=args.qpoints, npts=args.npts, width=args.width
        )
    else:
        dos = None

    results_dict = {"band_structure": band_structure, "dos": dos}
    output_name = (
        args.output_name
        if args.output_name.endswith(".pkl")
        else args.output_name + ".pkl"
    )
    with open(output_name, "wb") as f:
        pickle.dump(results_dict, f)

    if args.plot:
        fig, axes = plt.subplot_mosaic([["A", "A", "A", "B"]], figsize=(11, 8))
        bandax, dosax = axes["A"], axes["B"]
        emax = None

        if band_structure is not None:
            emax = np.max(band_structure._energies) + 50
            band_structure.plot(ax=bandax, emin=-50, emax=emax)
            bandax.set_ylabel("Energies (cm-1)")
        if dos is not None:
            dosax.fill_between(dos.get_weights(), dos.get_energies())
            emax = emax if emax else np.max(dos.get_energies())
            dosax.set_ylim(-50, emax)
            dosax.set_yticks([])
            dosax.set_xticks([])
            dosax.set_xlabel("DOS", fontsize=18)
        fig.tight_layout()
        plt.show()


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parent_calculation_parser = argparse.ArgumentParser(add_help=False)
    parent_calculation_parser.add_argument(
        "model", help="Path to the model used to generate the phonon dos."
    )
    parent_calculation_parser.add_argument(
        "posinp", help="Path to position file containing the equilibrium positions."
    )
    parent_calculation_parser.add_argument(
        "--device", default="cpu", help="Either 'cuda' or 'cpu'."
    )
    parent_calculation_parser.add_argument(
        "--supercell",
        help="Size of the supercell.",
        nargs=3,
        default=[6, 6, 6],
        type=int,
    )
    parent_calculation_parser.add_argument(
        "--qpoints",
        help="Qpoints grid for the dos estimation.",
        nargs=3,
        type=int,
        default=[30, 30, 30],
    )
    parent_calculation_parser.add_argument(
        "output_name", help="Name of the written file.", type=str
    )
    parent_calculation_parser.add_argument(
        "--plot",
        action="store_true",
        help="If used a plot of the results will be shown.",
    )

    parent_dos_parser = argparse.ArgumentParser(add_help=False)
    parent_dos_parser.add_argument(
        "--npts", help="Resolution in energy for the dos.", default=1000, type=int
    )
    parent_dos_parser.add_argument(
        "--width", help="Gaussian smearing to build the dos.", default=0.004, type=float
    )

    parent_band_parser = argparse.ArgumentParser(add_help=False)
    parent_band_parser.add_argument(
        "path", help="Path in the Brillouin Zone (for example 'GMKG' for graphene)."
    )

    calculation_type_parser = parser.add_subparsers(
        help="Calulation type choice.", dest="type"
    )
    dos_parser = calculation_type_parser.add_parser(
        "dos", parents=[parent_calculation_parser, parent_dos_parser]
    )
    band_parser = calculation_type_parser.add_parser(
        "band", parents=[parent_calculation_parser, parent_band_parser]
    )
    full_parser = calculation_type_parser.add_parser(
        "full",
        parents=[parent_calculation_parser, parent_band_parser, parent_dos_parser],
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
