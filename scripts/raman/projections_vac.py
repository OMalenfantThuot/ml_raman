#! /usr/bin/env python

import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.optimize as opt
import argparse
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from ml_raman.raman.gamma_modes_pristine import gamma_modes_pristine
from ml_raman.raman.dosdata import GeneralDOSData
import csv


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("natoms", help="Number on atoms in supercell.", type=int)
    parser.add_argument(
        "phonons_path", help="path to file containing phonons.", type=str
    )
    parser.add_argument(
        "--smearing",
        help="type of smearing. Either Gauss or Lorentz. Default is Gauss.",
        type=str,
        default="Gauss",
    )
    parser.add_argument(
        "width",
        help="smearing width. For Gaussian smearing, this c orresponds to \sigma. For Lorentzian smearing, this corresponds to \gamma.",
        type=float,
    )
    parser.add_argument(
        "--save_proj",
        help="create a savefile for raman mean projections, with smearing.",
        default=False,
    )
    parser.add_argument("npts", help="number of sampled points.", type=int)
    parser.add_argument(
        "output_name", help="name of the written figure file.", type=str
    )
    parser.add_argument(
        "--figure_title",
        help="Title of the raman spectra figure.",
        default="Raman spectra.",
    )
    parser.add_argument(
        "--save_fit",
        help="saves amplitude, the full width at half maximum and center of raman spectrum.",
        default=False,
    )
    parser.add_argument("--vac_idx", help="indices of vacancies.", type=int, nargs='*', default=None)
    return parser


def main(args):
    natoms = args.natoms
    npts = args.npts
    width = args.width
    smearing = args.smearing
    savename = (
        args.output_name
        if args.output_name.endswith(".jpg")
        else args.output_name + ".jpg"
    )
    nmodes = 3 * natoms
    eig = np.empty((2, nmodes))
    eig_avg = np.zeros((2, nmodes))
    raman_proj = np.empty((2, npts))

    # Define two gamma modes of pristine graphene
    if args.vac_idx is not None:
        vac_idx = args.vac_idx
        vac_idx.sort(reverse=True)
        l = len(vac_idx)
        v1, v2 = gamma_modes_pristine(natoms + l)
        for i in args.vac_idx:
            v1 = np.delete(
            v1, [3 * i, 3 * i + 1, 3 * i + 2])
            v2 = np.delete(
            v2, [3 * i, 3 * i + 1, 3 * i + 2])

            
    else:
        v1, v2 = gamma_modes_pristine(natoms)
    print(len(v1))

    f1 = h5py.File(args.phonons_path, "r")
    key_energies = list(f1.keys())[0]
    key_modes = list(f1.keys())[1]
    # Get the data
    eigenval =  np.array(f1[key_energies])
    eigvec =  np.array(f1[key_modes])
    abs1 = np.inner(v1, np.transpose(eigvec))
    abs2 = np.inner(v2, np.transpose(eigvec))
    raman = abs1**2 + abs2**2
    eig_avg[0] = eigenval
    eig_avg[1] = raman
    for el in list(zip(eigenval, np.transpose(eigvec),raman)):
        if el[2]>=10**(-4):
            print("projections",el)
    eig_raman_sorted = np.vstack((eigenval, raman))[:, eigenval.argsort()]
    eig_raman_sorted_tot = []
    eig_raman_sorted_tot.append(eig_raman_sorted)
    print("sum of raman projections on first mode = ", np.sum(abs1**2))
    print("sum of raman projections on second mode = ", np.sum(abs2**2))
    print("sum of raman projections = ", np.sum(raman))
    rdos = GeneralDOSData(eig_avg[0], eig_avg[1], info={"label": "raman"})
    rfig = plt.figure()
    # rdosax = rfig.add_axes([0.5, 0.2, 0.35, 0.7])
    rdosax = rfig.add_axes([0.2, 0.2, 0.75, 0.7])
    if width == 0:
        rdosax.plot(eig_avg[0], eig_avg[1], label="raman")
        y_max = max( eig_avg[1])

    else:
        # rdosax = rdos.plot(npts=npts, width=width, ax=rdosax, xmin=1200, xmax=1800)
        rdos.plot(npts=npts, width=width, ax=rdosax, smearing=smearing)
        rdosplot = rdos.sample_grid(
                npts=npts, width=width, xmin=1200, xmax=1800, smearing=smearing
            )
        y_max = max(rdosplot.get_weights())

    rdosax.tick_params(axis="both", which="major", labelsize=12)
    rdosax.set_xlim(0, 2000)
    rdosax.set_ylim(0, y_max)
    print("max = ", y_max)
    #rdosax.set_ylim(0, max(eig_avg[1]))
    rdosax.set_title(args.figure_title, fontsize=8)
    rdosax.set_xlabel("Frequency $\mathregular{(cm^{-1}}$)", fontsize=16)
    rdosax.set_ylabel("Intensity (a.u)", fontsize=16)

    savefile = args.save_proj
    if savefile:
        if width == 0:
            np.savetxt("raman" + args.output_name + ".dat", np.transpose(eig_avg))
        else:
            rdosplot = rdos.sample_grid(
                npts=npts, width=width, xmin=800, xmax=1800, smearing=smearing
            )
            raman_proj[1] = rdosplot.get_weights()
            raman_proj[0] = rdosplot.get_energies()
            np.savetxt("raman" + args.output_name + ".dat", np.transpose(raman_proj))

    save_fit = args.save_fit
    if save_fit and width != 0:
        if smearing == "Lorentz":
            # def _2Lorentzian(x, amp1, cen1, wid1, amp2,cen2,wid2):
            # return (amp1*wid1**2/((x-cen1)**2+wid1**2)) +\
            #           (amp2*wid2**2/((x-cen2)**2+wid2**2))
            def _1Lorentzian(x, amp1, cen1, wid1):
                return amp1 * wid1**2 / ((x - cen1) ** 2 + wid1**2)

            popt_1lorentzian_total = []

            for i in range(len(eig_raman_sorted_tot)):
                rdos_i = GeneralDOSData(
                    eig_raman_sorted_tot[i][0],
                    eig_raman_sorted_tot[i][1],
                    info={"label": "raman"},
                )
                rdosplot_i = rdos_i.sample_grid(
                    npts=npts, width=width, xmin=1200, xmax=1800, smearing=smearing
                )
                raman_weights_i = rdosplot_i.get_weights()
                raman_energies_i = rdosplot_i.get_energies()
                raman_select_weights_i = raman_weights_i[
                    (1250 <= raman_energies_i) & (raman_energies_i <= 1600)
                ]
                raman_select_energies_i = raman_energies_i[
                    (1250 <= raman_energies_i) & (raman_energies_i <= 1600)
                ]
                popt_1lorentzian_i, _ = curve_fit(
                    _1Lorentzian,
                    raman_select_energies_i,
                    raman_select_weights_i,
                    p0=[1, 1555, 12],
                )
                popt_1lorentzian_total.append(popt_1lorentzian_i)

            popt_1lorentzian_mean = np.mean(popt_1lorentzian_total, axis=0)
            popt_1lorentzian_std = np.std(popt_1lorentzian_total, axis=0)
            # with open(args.output_name+'_FWHM.txt', 'w') as f:
            #   f.write('{} {}'.format(iso_conc, FWHM))
            with open(
                "lorentzian_fit_" + args.output_name + ".csv", "w", newline=""
            ) as file:
                writer = csv.writer(file)
                field = [
                    "amplitude",
                    "center",
                    "FWHM",
                    "amplitude_err",
                    "center_err",
                    "FWHM_err",
                ]
                writer.writerow(field)
                amplitude = popt_1lorentzian_mean[0]
                center = popt_1lorentzian_mean[1]
                FWHM = 2 * np.abs(popt_1lorentzian_mean[2])
                amplitude_err = popt_1lorentzian_std[0]
                center_err = popt_1lorentzian_std[1]
                FWHM_err = 2 * popt_1lorentzian_std[2]
                writer.writerow(
                    [amplitude, center, FWHM, amplitude_err, center_err, FWHM_err]
                )

            print("Lorentzian amplitude : {} ± {}".format(amplitude, amplitude_err))
            print("Lorentzian center : {} ± {}".format(center, center_err))
            print("Lorentzian FWHM : {} ± {}".format(FWHM, FWHM_err))
            rdosplot = rdos.sample_grid(
                npts=npts, width=width, xmin=1200, xmax=1800, smearing=smearing
            )
            rdosax.set_ylim(0, max(rdosplot.get_weights()))
            print("max = ", max(rdosplot.get_weights()))
            # rdosax.set_ylim(0,1)
            # rdosax.set_ylim(0,0.0001)
            rdosax.plot(
                np.linspace(1250, 1800, 100),
                _1Lorentzian(np.linspace(1250, 1800, 100), *popt_1lorentzian_mean),
                label="lorentzian",
            )

        elif smearing == "Gauss":
            print("True")

            def _1Gaussian(x, amp1, cen1, sigma1):
                return (
                    amp1
                    * (1 / (sigma1 * (np.sqrt(2 * np.pi))))
                    * np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2))
                )

            popt_1gaussian_total = []
            for i in range(len(eig_raman_sorted_tot)):
                rdos_i = GeneralDOSData(
                    eig_raman_sorted_tot[i][0],
                    eig_raman_sorted_tot[i][1],
                    info={"label": "raman"},
                )
                rdosplot_i = rdos_i.sample_grid(
                    npts=npts, width=width, xmin=1200, xmax=1800, smearing=smearing
                )
                raman_weights_i = rdosplot_i.get_weights()
                raman_energies_i = rdosplot_i.get_energies()
                raman_select_weights_i = raman_weights_i[
                    (1250 <= raman_energies_i) & (raman_energies_i <= 1600)
                ]
                raman_select_energies_i = raman_energies_i[
                    (1250 <= raman_energies_i) & (raman_energies_i <= 1600)
                ]

                popt_1gaussian_i, _ = curve_fit(
                    _1Gaussian,
                    raman_select_energies_i,
                    raman_select_weights_i,
                    p0=[1, 1555, 12],
                )
                popt_1gaussian_total.append(popt_1gaussian_i)
            # with open(args.output_name+'_FWHM.txt', 'w') as f:
            #   f.write('{} {}'.format(iso_conc, FWHM))
            popt_1gaussian_mean = np.mean(popt_1gaussian_total, axis=0)
            popt_1gaussian_std = np.std(popt_1gaussian_total, axis=0)
            with open(
                "gaussian_fit_" + args.output_name + ".csv", "w", newline=""
            ) as file:
                writer = csv.writer(file)
                field = [
                    "amplitude",
                    "center",
                    "FWHM",
                    "amplitude_err",
                    "center_err",
                    "FWHM_err",
                ]
                writer.writerow(field)
                amplitude = popt_1gaussian_mean[0]
                center = popt_1gaussian_mean[1]
                FWHM = 2 * np.sqrt(2 * np.log(2)) * np.abs(popt_1gaussian_mean[2])
                amplitude_err = popt_1gaussian_std[0]
                center_err = popt_1gaussian_std[1]
                FWHM_err = 2 * np.sqrt(2 * np.log(2)) * popt_1gaussian_std[2]
                writer.writerow(
                    [amplitude, center, FWHM, amplitude_err, center_err, FWHM_err]
                )

            print("Gaussian amplitude: {} ± {}".format(amplitude, amplitude_err))
            print("Gaussian center: {} ± {}".format(center, center_err))
            print("Gaussian FWHM: {} ± {}".format(FWHM, FWHM_err))
            # rdosax.set_ylim(0,max(raman_weights))
            # rdosax.set_ylim(0,1)
            rdosplot = rdos.sample_grid(npts=npts, width=width, xmin=1200, xmax=1800)
            rdosax.set_ylim(0, max(rdosplot.get_weights()))
            rdosax.plot(
                np.linspace(1250, 1800, 100),
                _1Gaussian(np.linspace(1250, 1800, 100), *popt_1gaussian_mean),
                label="gaussian",
            )

    plt.legend()
    plt.savefig(savename)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
