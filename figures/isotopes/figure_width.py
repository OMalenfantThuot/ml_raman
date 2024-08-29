import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os

font = {"family": "CMU Serif", "size": 18}
plt.rc("font", **font)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")

datadir = "data/"

exp_width_data = pd.read_csv(os.path.join(datadir, "FWHM_exp.csv"))
concentrations = np.linspace(0, 1, 21)

data_dict = {
    "centers": [],
    "centers_std": [],
    "widths": [],
    "widths_std": [],
    "concentrations": [],
}
for concentration in concentrations:
    label = f"{concentration:3.2f}"
    avail, i = True, 1
    centers, widths = [], []
    while avail:
        try:
            data = pd.read_csv(
                os.path.join(datadir, f"Lorentz_fit_con_{label}_{i}.csv")
            )
            centers.append(data["center"])
            widths.append(data["FWHM"])
            i += 1
        except FileNotFoundError:
            avail = False

    if len(centers) > 0:
        data_dict["centers"].append(np.mean(centers))
        data_dict["centers_std"].append(np.std(centers))
        data_dict["widths"].append(np.mean(widths))
        data_dict["widths_std"].append(np.std(widths))
        data_dict["concentrations"].append(concentration)

fig = plt.figure(figsize=(8, 9))
gs = fig.add_gridspec(nrows=1, ncols=1, left=0.11, right=0.99, top=0.99, bottom=0.08)

ax = fig.add_subplot(gs[0, 0])
ax.grid("on")
exp = ax.errorbar(
    exp_width_data["concentration"],
    exp_width_data["width"],
    xerr=exp_width_data["concentration_err"],
    yerr=exp_width_data["width_err"],
    color="r",
    marker="o",
    markersize=5,
    linestyle="",
)
ml = ax.errorbar(
    data_dict["concentrations"],
    data_dict["widths"],
    yerr=data_dict["widths_std"],
    color="b",
    marker="o",
    linestyle="",
)


ax.legend(
    [exp, ml],
    [
        "Experimental",
        "ML",
    ],
    loc="lower center",
)
ax.set_xlabel(r"$^{13}$C concentration", fontsize=20)
ax.set_ylabel(r"G peak width ($\textrm{cm}^{-1}$)", fontsize=20)

# plt.show()
plt.savefig("figure_width.pdf")
