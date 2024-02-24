import csv
import pandas as pd
import numpy as np
from typing import Callable, Union, List, Tuple


def csv_append_line(
    fileName: str, data: Union[np.ndarray, List[Union[str, int, float]]]
):
    with open(fileName, "a", newline="") as csvfile:
        csv_writer = csv.writer(
            csvfile,
            delimiter=",",
            quoting=csv.QUOTE_MINIMAL,
        )
        csv_writer.writerow(data)


def csv_append_data(
    fileName: str,
    x: Union[int, float, str],
    y: Union[np.ndarray, List[Union[str, int, float]]],
):
    # using x,y[0],y[1],... append a line
    csv_append_line(fileName, [x, *y])


def _simple_load_csv_data(fileName: str, key_x: str, key_y: str):
    data = pd.read_csv(fileName)
    datax = data[key_x].to_numpy()
    datay = data[key_y].to_numpy()
    return datax, datay


def load_csv_data_with_stat(
    fileName: str,
    key_x: str,
    key_y: str,
    select_func: Union[Callable, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each unique x, calculate mean and std of y
    - fileName: file name
    - key_x, key_y: column names
    - select_func: function to select data (optional)
    """
    datax, datay = _simple_load_csv_data(fileName, key_x, key_y)
    #
    if select_func is not None:
        select_mask = select_func(datax, datay)
        datax = datax[select_mask]
        datay = datay[select_mask]
    #
    x_set = sorted(list(set(datax)))
    mean_y_list = []
    std_y_list = []
    for x in x_set:
        x_mask = datax == x
        y_x = datay[x_mask]
        mean_y = np.mean(y_x)
        std_y = np.std(y_x)
        mean_y_list.append(mean_y)
        std_y_list.append(std_y)

    return np.array(x_set), np.array(mean_y_list), np.array(std_y_list)


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5])
    csv_append_line("data.csv", a)
