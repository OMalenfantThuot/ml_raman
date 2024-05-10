#! /usr/bin/env python

from mlcalcdriver import Job, Posinp
from  mlcalcdriver.calculators import SchnetPackCalculator
from ml_raman.phonons.phonons_lanczos import PhononFromHessianLanczos
import argparse
import h5py
import numpy as np


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

    phonon_parser = calculation_type_subparser.add_parser(
        "phonon", parents=[parent_parser]
    )
    phonon_parser.add_argument(
        "--hessian_path", help="Path to the saved hessian matrix."
    )
    phonon_parser.add_argument(
        "--results_savepath", help="Name of the output file for the phonon results."
    )

    diag_mode_parser = phonon_parser.add_subparsers(
        help="Diagonalization mode choice", dest="diag_mode"
    )

    exact_diag_parser = diag_mode_parser.add_parser("exact")
    lanczos_diag_parser = diag_mode_parser.add_parser("lanczos")
    lanczos_diag_parser.add_argument("neigs",help="Number of highest eigenvalues computed by the lanczos method.",
    type=int
    )
    lanczos_diag_parser.add_argument("--initial_guess",help="first or second Gamma mode is used as a starting vector for iteration. Default: random.",
    type=int
    )
    return parser


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

        calculator = SchnetPackCalculator(args.model, device=args.device)

        job = Job(posinp=posinp, calculator=calculator)
        job.run("hessian")

        output_name = (
            args.hessian_savepath
            if args.hessian_savepath.endswith(".npy")
            else args.hessian_savepath + ".npy"
        )
        np.save(output_name, job.results["hessian"])

    elif args.type == "phonon":
        posinp = Posinp.from_file(args.posinp)
        hessian = np.load(args.hessian_path)
        diag_mode = args.diag_mode
        neigs = args.neigs
        initial_guess = args.initial_guess
        phonon = PhononFromHessianLanczos(posinp=posinp, hessian=hessian, diag_mode=diag_mode, neigs=neigs, initial_guess=initial_guess)
        phonon.run_exact_lanczos()

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
    parser = create_parser()
    args = parser.parse_args()
    main(args)
