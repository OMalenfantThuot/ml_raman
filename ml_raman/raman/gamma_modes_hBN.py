import numpy as np
import h5py
import os


def gamma_phonons_hBN(natoms_sc):
    nmodes = 3 * natoms_sc
    datapath = os.path.join(*__file__.split("/")[:-1], "modes", "gamma_modes_hBN.h5")
    datapath = "/" + datapath
    print(datapath)
    with h5py.File(datapath, "r") as file:
        v1_pr = np.array(file["v1"])
        v2_pr = np.array(file["v2"])
        v1 = np.tile(v1_pr, int(natoms_sc / 2))
        v2 = np.tile(v2_pr, int(natoms_sc / 2))
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return v1, v2


def gamma_eigendisplacements_hBN(natoms_sc):
    nmodes = 3 * natoms_sc
    datapath = os.path.join(*__file__.split("/")[:-1], "modes", "gamma_modes_hBN.h5")
    datapath = "/" + datapath
    with h5py.File(datapath, "r") as file:
        v1_pr = np.array(file["v1"])
        v2_pr = np.array(file["v2"])
        pristine_masses_pc = np.repeat(np.array([10.811, 14.00674]), 3)
        pristine_masses_sc = np.tile(pristine_masses_pc, int(natoms_sc / 2))
        v1 = np.tile(v1_pr, int(natoms_sc / 2))
        v2 = np.tile(v2_pr, int(natoms_sc / 2))
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)
        v1 = v1 / np.sqrt(pristine_masses_sc)
        v2 = v2 / np.sqrt(pristine_masses_sc)
    return v1, v2
