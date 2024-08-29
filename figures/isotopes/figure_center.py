import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.odr import ODR, Model, RealData
import os

font = {"family": "CMU Serif", "size": 18}
plt.rc("font", **font)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")

datadir = "data/"

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


def ml_linear_fit(x, a, b):
    return a * x + b


popt_ml, std_popt_ml = scipy.optimize.curve_fit(
    ml_linear_fit, data_dict["concentrations"], data_dict["centers"]
)


def exp_linear_fit(p, x):
    return p[0] * x + p[1]


exp_data = RealData(
    x=exp_center_data["concentration"],
    y=exp_center_data["center"],
    sx=exp_center_data["concentration_err"],
    sy=exp_center_data["center_err"],
)
model = Model(exp_linear_fit)
exp_odr = ODR(exp_data, model, beta0=[-60.0, 1500])
exp_odr.set_job(fit_type=2)
exp_fit = exp_odr.run()


x = np.linspace(0, 1, 100)

fig = plt.figure(figsize=(8, 9))
gs = fig.add_gridspec(nrows=1, ncols=1, left=0.13, right=0.99, top=0.99, bottom=0.08)

ax = fig.add_subplot(gs[0, 0])
ax.grid("on")
exp = ax.errorbar(
    exp_center_data["concentration"],
    exp_center_data["center"],
    xerr=exp_center_data["concentration_err"],
    yerr=exp_center_data["center_err"],
    color="r",
    marker="o",
    markersize=5,
    linestyle="",
)
exp_fit_curve = ax.plot(
    x,
    exp_linear_fit(exp_fit.beta, x),
    color="r",
    linestyle="--",
    label=r"Slope $(3 \pm 1 )$",
)
ml_fit_curve = ax.plot(
    x,
    ml_linear_fit(x, *popt_ml),
    color="b",
    linestyle="--",
    label=r"Slope $(3 \pm 1 )$",
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
    [exp, exp_fit_curve[0], ml_fit_curve[0], ml[0]],
    [
        "Experimental",
        "Slope: "
        + f"({exp_fit.beta[0]:3.1f} "
        + r"$\pm$"
        + f" {exp_fit.sd_beta[0]:3.1f})"
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

# plt.show()
plt.savefig("figure_center.pdf")
