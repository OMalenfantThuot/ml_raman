import numpy as np
import h5py
import argparse
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from mlcalcdriver import Posinp
from ase.io import read
from ml_raman.raman.dosdata import GeneralDOSData
import csv


def lorentzian(x, amp, cen, wid):
    return amp * wid**2 / ((x - cen) ** 2 + wid**2)


def gaussian(x, amp, cen, sigma):
    return (
        amp
        * (1 / (sigma * (np.sqrt(2 * np.pi))))
        * np.exp((-1.0 / 2.0) * (((x - cen) / sigma) ** 2))
    )


class Raman:
    def __init__(
        self, v1, v2, sc_path, phonons_path, width, smearing, npts, scale_masses
    ):
        self.v1 = v1
        self.v2 = v2
        self.sc_path = sc_path
        self.phonons_path = phonons_path
        self.width = width
        self.smearing = smearing
        self.npts = npts
        self.scale_masses = scale_masses

    def compute_phonon_projections(self):
        with h5py.File(self.phonons_path, "r") as f1:
            eigenval = np.array(f1[list(f1.keys())[0]])
            eigvec = np.array(f1[list(f1.keys())[1]])
        abs1 = np.inner(self.v1, np.transpose(eigvec))
        abs2 = np.inner(self.v2, np.transpose(eigvec))
        phonon_proj = abs1**2 + abs2**2
        return eigenval, phonon_proj

    def compute_eigendisplacement_projections(self):
        with h5py.File(self.phonons_path, "r") as f1:
            eigenval = np.array(f1[list(f1.keys())[0]])
            eigvec = np.array(f1[list(f1.keys())[1]])

        try:
            sc = Posinp.from_file(self.sc_path)
            defective_masses = np.repeat(sc.masses, 3)
        except Exception as e1:
            try:
                sc = read(self.sc_path)
                defective_masses = np.repeat(sc.get_masses(), 3)
            except Exception as e2:
                print(f"ML_Calc_driver exception: {str(e1)}")
                print(f"ASE exception: {str(e2)}")

        eigvec = np.transpose(eigvec) / np.sqrt(defective_masses)
        vec_norms = np.linalg.norm(eigvec, axis=1)

        # Check for invalid modes
        valid_modes = np.where(vec_norms > 0.01)
        eigvec = eigvec[valid_modes] / vec_norms[valid_modes].reshape(-1, 1)
        eigenval = eigenval[valid_modes]
        del sc, defective_masses, valid_modes

        abs1 = np.inner(self.v1, eigvec)
        abs2 = np.inner(self.v2, eigvec)
        print("v1 total projection: ", np.sum(abs1**2))
        print("v2 total projection: ", np.sum(abs2**2))
        eigendisplacement_proj = abs1**2 + abs2**2
        return eigenval, eigendisplacement_proj

    def compute_raman(self, eigenval, proj):
        if self.width != 0:
            rdos = GeneralDOSData(eigenval, proj, info={"label": "raman"})
            rdos_smeared = rdos.sample_grid(
                npts=self.npts,
                width=self.width,
                xmin=0,
                xmax=1800,
                smearing=self.smearing,
            )
            self.rdos = rdos
            self.eigenval = rdos_smeared.get_energies()
            self.raman = rdos_smeared.get_weights()
        else:
            self.eigenval = eigenval
            self.raman = proj

    def save_raman(self, output_name):
        raman_savename = (
            output_name if output_name.endswith(".dat") else output_name + ".dat"
        )
        np.savetxt("raman" + raman_savename, np.transpose([self.eigenval, self.raman]))

    def plot_raman(self, x_min, x_max, figure_title):
        plt.figure()
        if self.width == 0:
            plt.plot(self.eigenval, self.raman, label="raman")
            y_max = max(self.raman)
        else:
            rdosax = plt.gca()
            self.rdos.plot(
                npts=self.npts, width=self.width, ax=rdosax, smearing=self.smearing
            )
            y_max = max(self.raman)
        # Customize plot
        plt.gca().tick_params(axis="both", which="major", labelsize=10)
        plt.xlim(x_min, x_max)
        plt.ylim(0, y_max)
        plt.title(figure_title, fontsize=12)
        plt.xlabel("Frequency $\mathregular{(cm^{-1}}$)", fontsize=10)
        plt.ylabel("Intensity (a.u.)", fontsize=10)

    def compute_fit(self, output_name):
        if self.smearing == "Gauss" and self.width != 0:
            popt, _ = curve_fit(
                gaussian,
                self.eigenval[(1250 <= self.eigenval) & (self.eigenval <= 1600)],
                self.raman[(1250 <= self.eigenval) & (self.eigenval <= 1600)],
                p0=[1, 1555, 12],
            )
            self.amplitude, self.center, self.FWHM = (
                popt[0],
                popt[1],
                2 * np.sqrt(2 * np.log(2)) * np.abs(popt[2]),
            )
            self.amplitude_err, self.center_err, self.FWHM_err = (
                0,
                0,
                0,
            )  # to be implemented when I add averages
        elif self.smearing == "Lorentz" and self.width != 0:
            popt, _ = curve_fit(
                lorentzian,
                self.eigenval[(1250 <= self.eigenval) & (self.eigenval <= 1600)],
                self.raman[(1250 <= self.eigenval) & (self.eigenval <= 1600)],
                p0=[1, 1555, 12],
            )
            self.amplitude, self.center, self.FWHM = (
                popt[0],
                popt[1],
                2 * np.abs(popt[2]),
            )
            self.amplitude_err, self.center_err, self.FWHM_err = (
                0,
                0,
                0,
            )  # to be implemented when I add averages
        else:
            raise ValueError("Fit function not implemented.")
        fit_savename = (
            output_name if output_name.endswith(".csv") else output_name + ".csv"
        )
        with open(self.smearing + "_fit_" + fit_savename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "amplitude",
                    "center",
                    "FWHM",
                    "amplitude_err",
                    "center_err",
                    "FWHM_err",
                ]
            )
            writer.writerow(
                [
                    self.amplitude,
                    self.center,
                    self.FWHM,
                    self.amplitude_err,
                    self.center_err,
                    self.FWHM_err,
                ]
            )

    def plot_fit(self):
        if self.smearing == "Gauss":
            plt.plot(
                np.linspace(1250, 1800, 100),
                gaussian(
                    np.linspace(1250, 1800, 100), self.amplitude, self.center, self.FWHM
                ),
                label="Gaussian",
            )
        elif self.smearing == "Lorentz":
            plt.plot(
                np.linspace(1250, 1800, 100),
                lorentzian(
                    np.linspace(1250, 1800, 100),
                    self.amplitude,
                    self.center,
                    self.FWHM / 2,
                ),
                label="Lorentzian",
            )
        else:
            raise ValueError("Fit function not implemented.")
