import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


def draw_grid(
    ax: plt.Axes, N: int, x0: float, x1: float, y0: float, y1: float, **kwargs
) -> plt.Axes:
    """
    Draw a grid on the current axis.
    - ax: axis to draw
    - N: number of grid (N>2)
    - x0, x1: x-axis limits
    - y0, y1: y-axis limits
    """
    assert N > 2, "N should be greater than 2"
    x = np.linspace(x0, x1, N + 1)
    x_c = x[1:-1]
    y = np.linspace(y0, y1, N + 1)
    y_c = y[1:-1]
    #
    for i in range(N - 1):
        line = mlines.Line2D(
            [x_c[i], x_c[i]],
            [y0, y1],
            color="k",
            transform=ax.transAxes,
            **kwargs,
        )
        ax.add_line(line)
        #
        line = mlines.Line2D(
            [x0, x1],
            [y_c[i], y_c[i]],
            color="k",
            transform=ax.transAxes,
            **kwargs,
        )
        ax.add_line(line)
    return ax


def add_title(
    ax: plt.Axes, text: str, REL_X: float = -0.24, REL_Y: float = 0.96, **kwargs
) -> None:
    """
    Add a title to the current axis.
    - kwargs: fontsize = 14, color = "black", ...
    """
    fontsize = kwargs.pop("fontsize", 14)
    ax.annotate(
        text,
        xy=(REL_X, REL_Y),
        xycoords="axes fraction",
        fontsize=fontsize,
        fontweight="bold",
        **kwargs,
    )


if __name__ == "__main__":
    fig, ax = plt.subplots()
    draw_grid(ax, 5, 0, 1, 0, 1)
    add_title(ax, "Grid", fontsize=16, color="blue")
    plt.show()
