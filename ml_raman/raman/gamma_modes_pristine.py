import numpy as np
import h5py
import os


def gamma_modes_pristine(natoms_sc, return_frequency=False):
    nmodes = 3 * natoms_sc
    datapath = os.path.join(
        *__file__.split("/")[:-1], "modes", "gamma_modes_pristine.h5"
    )
    datapath = "/" + datapath

    with h5py.File(datapath, "r") as file:
        v1_pr = np.array(file["v1"])
        v2_pr = np.array(file["v2"])
        freq = float(file["frequency"][0])
        v1 = np.tile(v1_pr, (int(natoms_sc / 2), 1))
        v2 = np.tile(v2_pr, (int(natoms_sc / 2), 1))
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    if return_frequency == False:
        return v1, v2
    else:
        return (v1, v2), freq
