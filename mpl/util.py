import matplotlib.pyplot as plt
import numpy as np
from typing import Union


def mpl_default_colors() -> list:
    """
    Return the default colors of matplotlib.
    """
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def circle_mask(
    N: int,
    R: Union[int, float, None] = None,
    xc: Union[int, float] = 0,
    yc: Union[int, float] = 0,
    dtype: type = np.float_,
) -> np.ndarray:
    """
    Create a N*N circular mask.
    - N: size of the mask
    - R: radius of the circle
    - xc, yc: center of the circle, relative to the center of the mask
    """
    if R is None:
        R = N / 2
    y, x = np.ogrid[0:N, 0:N]
    mask = np.array(
        (x - N / 2 - xc) ** 2 + (y - N / 2 - yc) ** 2 <= (R) ** 2, dtype=dtype
    )
    return mask


if __name__ == "__main__":
    print(mpl_default_colors())
