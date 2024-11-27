import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import os
import pathlib

font = {"family": "CMU Serif", "size": 18}
plt.rc("font", **font)
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{bm}")

datadir = "data/"

graphene_data = pd.read_csv(pathlib.Path(datadir) / "log_graphene.csv")
hBN_data = pd.read_csv(pathlib.Path(datadir) / "log_hBN.csv")

graphene_train_loss = graphene_data["Train loss"]
graphene_val_loss = graphene_data["Validation loss"]
hBN_train_loss = hBN_data["Train loss"]
hBN_val_loss = hBN_data["Validation loss"]


fig = plt.figure(figsize=(8, 7))
gs = fig.add_gridspec(nrows=1, ncols=1, left=0.12, right=0.99, top=0.99, bottom=0.10)

ax = fig.add_subplot(gs[0, 0])
ax.grid("on")
ax.set_yscale("log")
ax.plot(
    np.arange(len(graphene_val_loss)),
    graphene_val_loss,
    color="#0570b0",
    label="Graphene train loss",
)
ax.plot(
    np.arange(len(graphene_train_loss)),
    graphene_train_loss,
    color="#74a9cf",
    label="Graphene validation loss",
)
ax.plot(
    np.arange(len(hBN_val_loss)),
    hBN_val_loss,
    color="#fd8d3c",
    label="hBN validation loss",
)
ax.plot(
    np.arange(len(hBN_train_loss)),
    hBN_train_loss,
    color="#e31a1c",
    label="hBN train loss",
)

ax.set_xlabel("Epochs", fontsize=20)
ax.set_ylabel("Loss value", fontsize=20)

ax.legend(loc="upper right", fontsize=20)

# plt.show()
plt.savefig("figure_training.pdf")
