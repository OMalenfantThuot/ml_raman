import torch
import numpy as np


def schnet_memory_estimation_for_graphene(
    interactions, cutoff, n_neurons, prefactor=0.000031, bias=0.03
):
    return prefactor * interactions * cutoff * n_neurons + bias


def get_graphene_patches_grid(
    mem_per_atom, interactions, cutoff, supercell_size_x, supercell_size_y
):
    # total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    total_gpu_mem = 30000
    if supercell_size_x * supercell_size_y * 2 * mem_per_atom < total_gpu_mem:
        return np.array([1, 1, 1])
    else:
        target_num_atoms = int(0.95 * total_gpu_mem / mem_per_atom)
        print(target_num_atoms)
        prim_len = 2.46738

        buffer = (interactions * cutoff) / np.sin(np.radians(60)) / prim_len

        new_x, new_y = 1, 1
        full_num_atoms = target_num_atoms * 2

        while full_num_atoms > target_num_atoms:
            if supercell_size_x / new_x >= supercell_size_y / new_y:
                new_x += 1
            else:
                new_y += 1

            patches_area = (
                (supercell_size_x / new_x)
                * (supercell_size_y / new_y)
                * np.sin(np.radians(60))
            )
            patches_num_atoms = (
                (supercell_size_x / new_x) * (supercell_size_y / new_y) * 2
            )
            full_area = (
                (supercell_size_x / new_x + 2 * buffer)
                * (supercell_size_y / new_y + 2 * buffer)
                * np.sin(np.radians(60))
            )
            full_num_atoms = patches_num_atoms * full_area / patches_area
        return np.array([new_x, new_y, 1])
