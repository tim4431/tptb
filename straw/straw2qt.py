# <<< This file is intended to convert density matrix from Strawberry Fields to QuTiP, and vice versa. >>>
# <<< For Strawberry Fields, the density matrix is in the form of rho_{iijjkkll},
# while for QuTiP, it is rho_{ijkl,ijkl}. >>>
import numpy as np
import string
from itertools import cycle


def _create_einstr(N: int) -> str:
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    alternating_list = [x + y for x, y in zip(lowercase, cycle(uppercase))]
    alternating_string = "".join(alternating_list)  # 'aAbBcCdD'
    origin_index = alternating_string[: 2 * N]  # 'aAbBcCdD'
    out_str = lowercase[:N] + uppercase[:N]  # 'abcdABCD'
    return origin_index, out_str


def reshape_dm_einstr(N: int) -> str:
    """N=4: iijjkkll->ijklijkl"""
    origin_index, out_str = _create_einstr(N)
    einstr = origin_index + "->" + out_str  # 'aAbBcCdD->abcdABCD'
    return einstr


def inv_reshape_dm_einstr(N: int) -> str:
    """N=4: ijklijkl->iijjkkll"""
    origin_index, out_str = _create_einstr(N)
    einstr = out_str + "->" + origin_index  # 'abcdABCD->aAbBcCdD'
    return einstr


def straw2qt(state_dm: np.ndarray, cutoff_dim: int = 5, N: int = 4) -> np.ndarray:
    state_dm = np.einsum(reshape_dm_einstr(N), state_dm)  # rho_{i,j,k,l,i,j,k,l}
    state_dm = state_dm.reshape(cutoff_dim**N, cutoff_dim**N)  # rho_{ijkl, ijkl}
    return state_dm


def qt2straw(state_dm: np.ndarray, cutoff_dim: int = 5, N: int = 4) -> np.ndarray:
    shape_straw = tuple([cutoff_dim] * (2 * N))
    state_dm = state_dm.reshape(shape_straw)  # rho_{i,j,k,l,i,j,k,l}
    state_dm = np.einsum(inv_reshape_dm_einstr(N), state_dm)  # rho_{i,i,j,j,k,k,l,l}
    return state_dm


if __name__ == "__main__":
    from state_from_str import fock_state_from_str

    a = fock_state_from_str("0110", 3, t="tensor")
    print(a.shape)
    a = straw2qt(a, cutoff_dim=3, N=2)
    print(a.shape)
    #
    a = np.zeros((9, 9))
    a = qt2straw(a, cutoff_dim=3, N=2)
    print(a.shape)
