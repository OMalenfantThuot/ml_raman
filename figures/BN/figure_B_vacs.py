import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os

font = {"family": "CMU Serif", "size": 18}
plt.rc("font", **font)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")


datadir = "data/B/"

raman_data = []

concentrations = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.030, 0.035]
for concentration in concentrations:
    con_data = []
    con_string = f"{concentration:.3f}"
    data_files = os.listdir(os.path.join(datadir, con_string))
    for f in data_files:
        con_data.append(np.loadtxt(os.path.join(datadir, con_string, f)).T)
    con_data = np.stack(con_data).mean(0)
    con_data[1] /= np.max(con_data[1])
    raman_data.append(con_data)

raman_data = np.stack(raman_data)
offset = 35
colors = [
    "#313695",
    "#4575b4",
    "#74add1",
    "#abd9e9",
    "#fdae61",
    "#f46d43",
    "#d73027",
    "#a50026",
]


fig = plt.figure(figsize=(8, 9))
gs = fig.add_gridspec(nrows=1, ncols=1, left=0.06, right=0.99, top=0.99, bottom=0.09)

ax = fig.add_subplot(gs[0, 0])
ax.set_yticks([])
ax.set_xlim(0, 1700)
ax.set_ylabel("Relative Intensities", fontsize=25)
ax.set_xlabel(r"Frequencies (cm$^{-1}$)", fontsize=25)

for i, (concentration, color) in enumerate(zip(concentrations, colors)):
    ax.plot(
        raman_data[i][0],
        raman_data[i][1] + concentration * offset,
        label=f"{concentration*100:.1f}\%",
        color=color,
    )
ax.legend(loc="upper left", title="B vacancies\nconcentration")

# ax.plot(raman_data[0][0], raman_data[-1][1] - raman_data[0][1])

# plt.show()
plt.savefig("B_vacancies.pdf")
