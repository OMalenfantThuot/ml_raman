#! /usr/bin/env python

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from ml_raman.raman.gamma_modes_graphene import (
    gamma_phonons_graphene,
    gamma_eigendisplacements_graphene,
)
from ml_raman.raman.gamma_modes_hBN import (
    gamma_phonons_hBN,
    gamma_eigendisplacements_hBN,
)
from ml_raman.raman.dosdata import GeneralDOSData
from ml_raman.raman.raman import Raman
import csv

plt.style.use("ggplot")


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "primitive_cell",
        help="Primitive cell. Arguments supported for now are C and BN.",
        type=str,
    )
    parser.add_argument("natoms", help="Number of atoms in supercell.", type=int)
    parser.add_argument("sc_path", help="Path to file of supercell.", type=str)
    parser.add_argument(
        "phonons_path", help="Path to file containing phonons.", type=str
    )
    parser.add_argument(
        "width",
        help="smearing width. For Gaussian smearing, this corresponds to σ. For Lorentzian smearing, this corresponds to γ.",
        type=float,
    )
    parser.add_argument(
        "output_name", help="name of the written figure file.", type=str
    )
    parser.add_argument(
        "--vacancies_file", help="Path to the vacancies file", default=None
    )
    parser.add_argument(
        "--scale_masses",
        action=argparse.BooleanOptionalAction,
        help="option to scale the eigenvectors of dynamical matrix by masses to get eigendisplacements projections.",
        default=True,
    )
    parser.add_argument(
        "--smearing",
        help="type of smearing. Either Gauss or Lorentz. Default is Gauss.",
        type=str,
        default="Gauss",
    )
    parser.add_argument(
        "--npts", help="number of sampled points.", type=int, default=2000
    )
    parser.add_argument(
        "--figure_title",
        help="Title of the raman spectra figure.",
        default="Raman spectra",
    )
    parser.add_argument(
        "--figure_x_min",
        help="Minimum value for the figure x axis.",
        type=float,
        default=1000,
    )
    parser.add_argument(
        "--figure_x_max",
        help="Maximum value for the figure x axis.",
        type=float,
        default=1600,
    )
    parser.add_argument(
        "--save_raman",
        action=argparse.BooleanOptionalAction,
        help="create a savefile for raman projections, with smearing.",
        default=True,
    )
    parser.add_argument(
        "--save_fit",
        action=argparse.BooleanOptionalAction,
        help="Saves amplitude, the full width at half maximum and center of raman spectrum.",
        default=True,
    )
    return parser


def compute_gamma_phonons(pc, natoms, indices_to_remove):
    # function that constructs Raman active phonon modes at \Gamma from primitive cell modes.
    n_vac = len(indices_to_remove) if indices_to_remove is not None else 0
    if pc == "C":
        v1, v2 = gamma_phonons_graphene(natoms)
    elif pc == "BN":
        v1, v2 = gamma_phonons_hBN(natoms)
    else:
        raise ValueError("Primitive cell not supported by script.")
    if n_vac != 0:
        v1 = remove_vacancies(v1, indices_to_remove)
        v2 = remove_vacancies(v2, indices_to_remove)
    v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    return v1, v2


def compute_gamma_eigendisplacements(pc, natoms, indices_to_remove):
    # function that constructs Raman active eigendisplacement modes at \Gamma from primitive cell modes.
    n_vac = len(indices_to_remove) if indices_to_remove is not None else 0
    if pc == "C":
        v1, v2 = gamma_eigendisplacements_graphene(natoms)
    elif pc == "BN":
        v1, v2 = gamma_eigendisplacements_hBN(natoms)
    else:
        raise ValueError("Primitive cell not supported by script.")

    if n_vac != 0:
        v1 = remove_vacancies(v1, indices_to_remove)
        v2 = remove_vacancies(v2, indices_to_remove)
    v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    return v1, v2


def remove_vacancies(mode, indices_to_remove):
    # function that removes components from phonon/eigendisplacement vectors at vacancies' indicies.
    for idx in indices_to_remove:
        mode = np.delete(mode, [3 * idx, 3 * idx + 1, 3 * idx + 2])
    return mode


def main(args):
    pc = args.primitive_cell
    sc_path = args.sc_path
    phonons_path = args.phonons_path
    natoms = args.natoms
    npts = args.npts
    width = args.width
    smearing = args.smearing
    scale_masses = args.scale_masses
    fig_savename = (
        args.output_name
        if args.output_name.endswith(".jpg")
        else args.output_name + ".jpg"
    )
    x_min = args.figure_x_min
    x_max = args.figure_x_max
    figure_title = args.figure_title
    save_raman = args.save_raman
    save_fit = args.save_fit
    scale_masses = args.scale_masses
    nmodes = 3 * natoms

    if args.vacancies_file is not None:
        indices_to_remove = np.flip(np.sort(np.load(args.vacancies_file)))
    else:
        indices_to_remove = None

    # compute projections.
    if scale_masses:
        v1, v2 = compute_gamma_eigendisplacements(pc, natoms, indices_to_remove)
        # create a Raman instance.
        raman_inst = Raman(
            v1, v2, sc_path, phonons_path, width, smearing, npts, scale_masses
        )
        (
            eigenval,
            proj,
        ) = raman_inst.compute_eigendisplacement_projections()
    else:
        v1, v2 = compute_gamma_phonons(pc, natoms, indices_to_remove)
        # create a Raman instance.
        raman_inst = Raman(
            v1, v2, sc_path, phonons_path, width, smearing, npts, scale_masses
        )
        eigenval, proj = raman_inst.compute_phonon_projections()

    raman_inst.compute_raman(eigenval, proj)

    # plot raman spectrum.
    raman_inst.plot_raman(x_min, x_max, figure_title)

    if save_raman:
        raman_inst.save_raman(args.output_name)
    if save_fit and width != 0:
        raman_inst.compute_fit(args.output_name)
        raman_inst.plot_fit()
    plt.legend()
    plt.savefig(fig_savename)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
