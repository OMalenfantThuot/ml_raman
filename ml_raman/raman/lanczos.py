from mlcalcdriver import Posinp, Job
from mlcalcdriver.calculators import Calculator
from mlcalcdriver.globals import ANG_TO_B, B_TO_ANG, EV_TO_HA, HA_TO_CMM1, AMU_TO_EMU
import numpy as np
import numpy.typing as npt
from typing import Literal, Tuple, List
from copy import deepcopy
import scipy


def apply_dynamic_matrix(
    posinp: Posinp,
    calculator: Calculator,
    v: npt.NDArray[float],
    patches=None,
) -> npt.NDArray[float]:

    mass_vector = np.sqrt(np.array([atom.mass for atom in posinp for _ in range(3)]))
    mass_scaled_v = v / mass_vector
    mass_scaled_v_norm = np.linalg.norm(mass_scaled_v)

    if mass_scaled_v_norm == 0:
        return v
    else:
        moves, coeffs = get_finite_difference_coefficients(order=2)

        average_move_amplitude = 0.005
        move_normalization = np.linalg.norm(
            np.ones(len(posinp)) * average_move_amplitude
        )
        base_displacement_vector = (
            mass_scaled_v / mass_scaled_v_norm
        ) * move_normalization
        base_displacement_vector_norm = np.linalg.norm(base_displacement_vector)

        gradient_values = np.zeros((len(moves), len(posinp), 3), dtype=np.float32)
        for i, move in enumerate(moves):
            temp_posinp = deepcopy(posinp)

            displacement_vector = (move * base_displacement_vector).reshape(-1, 3)
            temp_posinp = temp_posinp.translate(displacement_vector)

            if patches is not None:
                temp_patches = deepcopy(patches)
                translated_subcells = []
                for subcell, copy_idx in zip(temp_patches[0], temp_patches[3]):
                    subcell.positions = (
                        subcell.positions + displacement_vector[copy_idx]
                    )
            else:
                temp_patches = None

            gradient_values[i, ...] = get_gradient(
                temp_posinp, calculator, patches=temp_patches
            )

        gradient_derivatives = (
            np.broadcast_to(coeffs[:, np.newaxis, np.newaxis], gradient_values.shape)
            * gradient_values
        ).sum(0).flatten() / base_displacement_vector_norm
        gradient_derivatives *= mass_scaled_v_norm
        gradient_derivatives /= mass_vector
        return gradient_derivatives


def get_gradient(
    posinp: Posinp, calculator: Calculator, patches=None
) -> npt.NDArray[np.float32]:
    calculator.patches = patches
    job = Job(posinp=posinp, calculator=calculator)
    job.run("forces")
    return -1 * job.results["forces"]


def get_finite_difference_coefficients(
    order: int = 2,
) -> Tuple[npt.NDArray[int], npt.NDArray[float]]:
    if order == 2:
        moves = [-1, 1]
        coeffs = [-0.5, 0.5]
    elif order == 4:
        moves = [-2, -1, 1, 2]
        coeffs = [1.0 / 12, -2.0 / 3, 2.0 / 3, -1.0 / 12]
    else:
        raise NotImplementedError()
    return (np.array(moves), np.array(coeffs))


def lanczos_raman_projections(
    posinp: Posinp,
    calculator: Calculator,
    material: Literal["graphene", "BN"],
    k: int,
    patches=None,
):
    if material == "graphene":
        from ml_raman.raman.gamma_modes_graphene import gamma_eigendisplacements_graphene

        v1, v2 = gamma_eigendisplacements_graphene(len(posinp))
        v1, v2 = v1.flatten(), v2.flatten()
    elif material == "BN":
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    n = 3 * len(posinp)
    import time

    def mv_mul(v_in):
        v_in = v_in.flatten()
        v_out = apply_dynamic_matrix(posinp, calculator, v_in, patches)
        return v_out

    operator = scipy.sparse.linalg.LinearOperator(shape=(n, n), matvec=mv_mul)
    t1 = time.time()
    eig, eigv = scipy.sparse.linalg.eigsh(
        operator, k=k, v0=v1, tol=1e-5, ncv=int(1.6 * k)
    )
    print("Timing scipy: ", time.time() - t1)
    eig *= EV_TO_HA * B_TO_ANG**2 / AMU_TO_EMU
    eig = np.sign(eig) * np.sqrt(np.abs(eig)) * HA_TO_CMM1
    return eig, eigv
