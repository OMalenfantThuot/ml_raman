#! /usr/bin/env python

from mlcalcdriver import Job, Posinp
from mlcalcdriver.calculators import PatchSPCalculator
from mlcalcdriver.workflows.phonon import PhononFromHessian
from ml_raman.raman.gamma_modes_pristine import gamma_modes_pristine
from schnetpack.utils import load_model
import argparse
import h5py
import numpy as np
import scipy


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "posinp", help="Path to position file containing the equilibrium positions."
    )
    parent_parser.add_argument("model", help="Path to the model.")
    parent_parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Either 'cuda' or 'cpu'.",
    )

    calculation_type_subparser = parser.add_subparsers(
        help="Calculation type choice.", dest="type"
    )
    hessian_parser = calculation_type_subparser.add_parser(
        "hessian", parents=[parent_parser]
    )
    hessian_parser.add_argument(
        "--hessian_savepath",
        default="hessian.npy",
        help="Name of the output file for the hessian matrix.",
    )
    hessian_parser.add_argument(
        "--grid",
        default=None,
        type=int,
        help="Size of the grid. If None, will be chosen by the code.",
    )
    hessian_parser.add_argument("--sparse", default=False, action="store_true")
    phonon_parser = calculation_type_subparser.add_parser(
        "phonon", parents=[parent_parser]
    )
    phonon_parser.add_argument(
        "--hessian_path", help="Path to the saved hessian matrix."
    )
    phonon_parser.add_argument(
        "--results_savepath", help="Name of the output file for the phonon results."
    )
    phonon_parser.add_argument(
        "--sparse",
        default=False,
        action="store_true",
        help="Use sparse matrices and Lanczos algorithm.",
    )
    phonon_parser.add_argument(
        "--use_jax",
        default=False,
        action="store_true",
        help="Use JAX for diagonalization",
    )
    phonon_parser.add_argument(
        "--eigs_proportion",
        type=float,
        default=0.05,
        help="Proportion of eigenvalues computed by Lanczos. Only relevant in sparse calculations.",
    )
    phonon_parser.add_argument(
        "--tol",
        type=float,
        default=0,
        help="Tolerance for the Lanczos convergence. Only relevent in sparse calculations.",
    )
    phonon_parser.add_argument(
        "--initial_guess",
        default=None,
        choices=[None, "graphene", "hBN"],
        help="Use the Gamma Raman active modes  as initial guess in the Lanczos algorithm.",
    )
    return parser


def load_hessian(args):
    path = args.hessian_path
    assert path.endswith(".npy") or path.endswith(".npz")

    if path.endswith(".npy"):
        hessian = np.load(path)
        if args.sparse:
            hessian = scipy.sparse.csr_array(hessian)
    else:
        hessian = scipy.sparse.load_npz(path)
        if not args.sparse:
            hessian = hessian.todense()[np.newaxis, ...]
    return hessian


def main(args):
    if args.type == "hessian":
        try:
            posinp = Posinp.from_file(args.posinp)
        except Exception as e:
            print(f"ML_Calc_Driver read exception: {str(e)}")
            try:
                posinp = Posinp.read(args.posinp)
            except Exception as e:
                print(f"ASE read exception: {str(e)}")
                exit()

        if args.grid is None:
            from utils.models import get_graphene_patches_grid

            size = int(np.sqrt(len(posinp) / 2))
            temp_model = load_model(args.model, map_location="cpu")
            n_neurons = temp_model.representation.embedding.weight.shape[1]
            n_interactions = len(temp_model.representation.interactions)
            cutoff = float(
                temp_model.representation.interactions[0].cutoff_network.cutoff
            )
            del temp_model

            patches_grid = get_graphene_patches_grid(
                "hessian", n_interactions, cutoff, n_neurons, size, size
            )
        else:
            patches_grid = np.array([args.grid, args.grid, 1])
        print("Patches grid: ", patches_grid)

        calculator = PatchSPCalculator(
            args.model,
            device=args.device,
            subgrid=patches_grid,
            sparse=args.sparse,
        )

        job = Job(posinp=posinp, calculator=calculator)
        job.run("hessian")

        if args.sparse:
            output_name = (
                args.hessian_savepath
                if args.hessian_savepath.endswith(".npz")
                else args.hessian_savepath + ".npz"
            )
            scipy.sparse.save_npz(output_name, job.results["hessian"])
        else:
            output_name = (
                args.hessian_savepath
                if args.hessian_savepath.endswith(".npy")
                else args.hessian_savepath + ".npy"
            )
            np.save(output_name, job.results["hessian"])

    elif args.type == "phonon":
        assert not (
            args.sparse and args.use_jax
        ), "Choose only one of sparse or JAX diagonalization."
        try:
            posinp = Posinp.from_file(args.posinp)
        except Exception as e:
            print(f"ML_Calc_Driver read exception: {str(e)}")
            try:
                posinp = Posinp.read(args.posinp)
            except Exception as e:
                print(f"ASE read exception: {str(e)}")
                exit()

        hessian = load_hessian(args)

        if args.sparse:
            if args.initial_guess is not None:
                if args.initial_guess == "graphene":
                    from mlcalcdriver.globals import (
                        AMU_TO_EMU,
                        EV_TO_HA,
                        HA_TO_CMM1,
                        B_TO_ANG,
                    )

                    (v1, v2), sigma = gamma_modes_pristine(
                        int(hessian.shape[0] / 3), return_frequency=True
                    )
                    sigma = (
                        (sigma / HA_TO_CMM1) ** 2
                        * AMU_TO_EMU
                        / (EV_TO_HA * B_TO_ANG**2)
                    )
                    v0 = v1
                else:
                    raise NotImplementedError()
                del v1, v2
            else:
                sigma, v0 = None, None
            sparse_kwargs = {
                "k": int(args.eigs_proportion * hessian.shape[0]),
                "sigma": sigma,
                "mode": "normal",
                "tol": args.tol,
                "v0": v0,
            }
        else:
            sparse_kwargs = {}

        phonon = PhononFromHessian(posinp=posinp, hessian=hessian, sparse=args.sparse)
        phonon.run(use_jax=args.use_jax, sparse_kwargs=sparse_kwargs)

        savename = (
            args.results_savepath
            if args.results_savepath.endswith(".h5")
            else args.results_savepath + ".h5"
        )
        with h5py.File(savename, "w") as f:
            f.create_dataset("modes", data=phonon.normal_modes)
            f.create_dataset("energies", data=phonon.energies)

    else:
        raise NotImplementedError()


if __name__ == "__main__":
    import time

    t1 = time.time()
    parser = create_parser()
    args = parser.parse_args()
    main(args)
    print("Timing without importations: ", time.time() - t1)
