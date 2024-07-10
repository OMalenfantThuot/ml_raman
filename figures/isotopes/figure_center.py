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
exp_center_data = pd.read_csv(os.path.join(datadir, "center_exp.csv"))

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


def linear_fit(x, a, b):
    return a * x + b


popt_exp, std_popt_exp = scipy.optimize.curve_fit(
    linear_fit, exp_center_data["concentration"], exp_center_data["center"]
)
popt_ml, std_popt_ml = scipy.optimize.curve_fit(
    linear_fit, data_dict["concentrations"], data_dict["centers"]
)
x = np.linspace(0, 1, 100)

fig = plt.figure(figsize=(8, 9))
gs = fig.add_gridspec(nrows=1, ncols=1, left=0.13, right=0.99, top=0.99, bottom=0.08)

ax = fig.add_subplot(gs[0, 0])
ax.grid("on")
exp = ax.errorbar(
    exp_center_data["concentration"],
    exp_center_data["center"],
    color="r",
    marker="o",
    linestyle="",
)
exp_fit = ax.plot(
    x, linear_fit(x, *popt_exp), color="r", linestyle="--", label=r"Slope $(3 \pm 1 )$"
)
ml_fit = ax.plot(
    x, linear_fit(x, *popt_ml), color="b", linestyle="--", label=r"Slope $(3 \pm 1 )$"
)
ml = ax.errorbar(
    data_dict["concentrations"],
    data_dict["centers"],
    yerr=data_dict["centers_std"],
    color="b",
    marker="o",
    linestyle="",
)


ax.legend(
    [exp[0], exp_fit[0], ml_fit[0], ml[0]],
    [
        "Experimental",
        "Slope: "
        + f"({popt_exp[0]:3.1f} "
        + r"$\pm$"
        + f" {np.sqrt(std_popt_exp[0,0]):3.1f})"
        + r"$\textrm{cm}^{-1}$",
        "Slope: "
        + f"({popt_ml[0]:3.1f} "
        + r"$\pm$"
        + f" {np.sqrt(std_popt_ml[0,0]):3.1f})"
        + r"$\textrm{cm}^{-1}$",
        "ML",
    ],
)
ax.set_xlabel(r"$^{13}$C concentration", fontsize=20)
ax.set_ylabel(r"G peak position ($\textrm{cm}^{-1}$)", fontsize=20)

#plt.show()
plt.savefig("figure_center.pdf")
