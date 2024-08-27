import numpy as np
import random
import argparse


# This scripts generates evenly spaced indices in BN supercell.
def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("num_atoms", help="total number of atoms.", type=int)
    parser.add_argument(
        "concentration",
        help="concentration of vacancies. Note that concentration = #vac/#atoms",
        type=float,
    )
    parser.add_argument(
        "atom_type",
        help="type of vacancies. 'B' for even indices, 'N' for odd indices.",
        type=str,
    )
    parser.add_argument(
        "--distribution",
        help="distribution type: 'even' for equally spaced  or 'random' for randomly distributed.",
        type=str,
        choices=["even", "random"],
        default="even",
    )
    return parser


def generate_indices(num_atoms, concentration, atom_type, distribution):
    """
    Generate a subset of indices for hBN that are equally spaced based on the desired concentration and atom type.

    :param num_atoms: Total number of atoms.
    :param concentration: Desired concentration of selected indices (0 < concentration <= 1).
    :param atom_type: Type of atoms to select ('B' for even indices, 'N' for odd indices).
    :return: A list of selected indices.
    """
    # Validate the atom_type
    if atom_type not in ["B", "N"]:
        raise ValueError(
            "atom_type must be either 'B' (even indices) or 'N' (odd indices)."
        )

    # Determine the starting index based on the atom type
    start_index = 0 if atom_type == "B" else 1

    # Generate a list of all valid indices for the specified atom type
    valid_indices = list(range(start_index, num_atoms, 2))

    # Calculate the number of indices to select
    num_selected = int(np.round(2 * concentration * len(valid_indices)))

    if distribution == "even":
        # Determine the step size to ensure equal spacing
        step_size = len(valid_indices) // num_selected

        # Generate equally spaced indices
        selected_indices = [valid_indices[i * step_size] for i in range(num_selected)]
    elif distribution == "random":
        selected_indices = random.sample(valid_indices, num_selected)
        # Sort the indices to be in increasing order
        selected_indices.sort()
    else:
        raise ValueError("Invalid distribution type. Choose 'even' or 'random'.")

    return selected_indices


def main(args):
    selected_indices = generate_indices(
        args.num_atoms, args.concentration, args.atom_type, args.distribution
    )
    print("selected indices : ")
    print(" ".join(map(str, selected_indices)))


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
