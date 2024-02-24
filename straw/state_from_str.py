import numpy as np
from typing import Callable


def _tensor_dot(*args) -> np.ndarray:
    """
    - Tensor product of input matrices
    - Return dimension: (d1, d2, ..., dn)
    """
    res = 1
    for op in args:
        res = np.tensordot(res, op, axes=0)
    return res


def _tensor_kron(*args) -> np.ndarray:
    """
    - Tensor product of input matrices
    - Return dimension: (d1*d2*...*dn)"""
    res = 1
    for op in args:
        res = np.kron(res, op)
    return res


def _tensor_func(*args, t: str) -> np.ndarray:
    if t == "tensor":
        return _tensor_dot(*args)
    elif t == "kron":
        return _tensor_kron(*args)
    else:
        raise ValueError("t must be tensor or kron")


def _fock_n(n: int, cutoff_dims: int) -> np.ndarray:
    """Fock state |n>"""
    _fock = np.zeros(cutoff_dims)
    _fock[n] = 1
    return _fock


def _mono_fock_state_from_str(
    state_str: str, cutoff_dims: int, t: str = "kron"
) -> np.ndarray:
    """
    - state_str: 00 / 012 / 123
    - cutoff_dims: int
    - t: "tensor" / "kron"
    """
    state_list = []
    for ch in state_str:
        state_list.append(_fock_n(int(ch), cutoff_dims))
    return _tensor_func(*state_list, t=t)


def _mono_qubit_state_from_str(
    state_str: str, cutoff_dims: int, t: str = "kron"
) -> np.ndarray:
    """
    - state_str: 0 / 01 / 0101
    - cutoff_dims: int
    - t: "tensor" / "kron"
    """
    #
    Fock_0 = _fock_n(0, cutoff_dims)
    Fock_1 = _fock_n(1, cutoff_dims)
    state_list = []
    for ch in state_str:
        if ch == "0":
            state_list.append(Fock_1)
            state_list.append(Fock_0)
        elif ch == "1":
            state_list.append(Fock_0)
            state_list.append(Fock_1)
        else:
            raise ValueError("qubit state_str must be 0 or 1")
    return _tensor_func(*state_list, t=t)


def state_from_str(
    mono_state_from_str_func: Callable,
    state_str: str,
    cutoff_dims: int,
    t: str = "kron",
) -> np.ndarray:
    """
    - mono_state_from_str_func: either fock state or qubit state
    - state_str: 01 / 00+11 / 00-01+11
    - cutoff_dims: int
    - t: "tensor" / "kron"
    """
    states_list = []
    current_number = ""
    sign = 1

    for char in state_str:
        if char.isdigit():
            current_number += char
        else:
            if current_number:
                states_list.append(
                    sign * mono_state_from_str_func(current_number, cutoff_dims, t)
                )
                current_number = ""
            if char == "+":
                sign = 1
            elif char == "-":
                sign = -1

    if current_number:
        states_list.append(
            sign * mono_state_from_str_func(current_number, cutoff_dims, t)
        )

    #
    return sum(states_list) / np.sqrt(len(states_list))


def qubit_state_from_str(
    state_str: str, cutoff_dims: int, t: str = "kron"
) -> np.ndarray:
    return state_from_str(_mono_qubit_state_from_str, state_str, cutoff_dims, t)


def fock_state_from_str(
    state_str: str, cutoff_dims: int, t: str = "kron"
) -> np.ndarray:
    return state_from_str(_mono_fock_state_from_str, state_str, cutoff_dims, t)


def bell_state(cutoff_dims: int, t: str = "kron") -> np.ndarray:
    """
    bell state: 00+11
    - t: "tensor" / "kron"
    """
    return qubit_state_from_str("00+11", cutoff_dims, t)


if __name__ == "__main__":
    bell_state_tensor = bell_state(4, t="tensor")
    print(bell_state_tensor.shape)
