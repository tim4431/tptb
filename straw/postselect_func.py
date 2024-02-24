import numpy as np


def postselect_qubit_2_2_ket(state_ket: np.ndarray, cutoff_dim=5) -> np.ndarray:
    """
    - postselect qubit 2 + 2
    - principle: 0,0,i,j -> 0; i,j,0,0 -> 0
    """

    def index(i, j, k, l):
        return i * cutoff_dim**3 + j * cutoff_dim**2 + k * cutoff_dim + l

    # postselect
    for i in range(cutoff_dim):
        for j in range(cutoff_dim):
            idx = index(i, j, 0, 0)
            state_ket[idx] = 0
            idx = index(0, 0, i, j)
            state_ket[idx] = 0
    return state_ket


def postselect_qubit_2_2_dm(state_dm: np.ndarray, cutoff_dim: int = 5) -> np.ndarray:
    """
    postselection of 2 qubits, density matrix approach
    - postselect qubit 2 + 2
    - principle: 0,0,i,j -> 0; i,j,0,0 -> 0
    """

    def index(i, j, k, l) -> int:
        return i * cutoff_dim**3 + j * cutoff_dim**2 + k * cutoff_dim + l

    # postselect
    for i in range(cutoff_dim):
        for j in range(cutoff_dim):
            idx = index(i, j, 0, 0)
            state_dm[idx, :] = 0
            state_dm[:, idx] = 0
            idx = index(0, 0, i, j)
            state_dm[idx, :] = 0
            state_dm[:, idx] = 0
    return state_dm


def postselect_nphoton_ge2_dm(state_dm: np.ndarray, cutoff_dim: int = 5) -> np.ndarray:
    """
    - postselect nphoton >= 2
    - principle: sum(i,j,k,l) < 2 -> 0
    """

    def index(i, j, k, l) -> int:
        return i * cutoff_dim**3 + j * cutoff_dim**2 + k * cutoff_dim + l

    # postselect
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if i + j + k + l < 2:
                        idx = index(i, j, k, l)
                        state_dm[idx, :] = 0
                        state_dm[:, idx] = 0
    return state_dm


def postselect_PBSCNOT_dm(state_dm: np.ndarray, cutoff_dim: int = 5) -> np.ndarray:
    def index(i, j, k, l, a, b, c, d) -> int:
        return (
            i * cutoff_dim**7
            + j * cutoff_dim**6
            + k * cutoff_dim**5
            + l * cutoff_dim**4
            + a * cutoff_dim**3
            + b * cutoff_dim**2
            + c * cutoff_dim
            + d
        )

    # postselect
    for x0 in range(cutoff_dim):
        for x1 in range(cutoff_dim):
            # # postselect
            # for i in range(cutoff_dim):
            #     for j in range(cutoff_dim):
            #         for h0 in range(cutoff_dim):
            #             for h1 in range(cutoff_dim):
            #                 idx = index(x0, x1, i, j, 0, 0, h0, h1)
            #                 state_dm[idx, :] = 0
            #                 state_dm[:, idx] = 0
            #                 idx = index(x0, x1, 0, 0, i, j, h0, h1)
            #                 state_dm[idx, :] = 0
            #                 state_dm[:, idx] = 0
            # herald
            for i in range(cutoff_dim):
                for j in range(cutoff_dim):
                    for k in range(cutoff_dim):
                        for l in range(cutoff_dim):
                            for h1 in range(cutoff_dim):
                                idx = index(x0, x1, i, j, k, l, 0, h1)
                                state_dm[idx, :] = 0
                                state_dm[:, idx] = 0
                            for h0 in range(cutoff_dim):
                                idx = index(x0, x1, i, j, k, l, h0, 0)
                                state_dm[idx, :] = 0
                                state_dm[:, idx] = 0
    return state_dm
