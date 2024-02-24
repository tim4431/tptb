import matplotlib.pyplot as plt
from typing import Callable
import time
import numpy as np
from scipy.optimize import minimize
from collections import deque


class MplInteractive:
    def __init__(self, callback_func: Callable, HIST_LENGTH: int = 50):
        self.datas = deque(maxlen=HIST_LENGTH)
        self.callback_func = callback_func

    def init_plot(self):
        self.fig, self.axs = plt.subplots()
        self.axs.set_xlabel("x")
        self.axs.set_ylabel("y")
        (self.plot1,) = self.axs.plot([], [])

    def start_session(self):
        plt.ion()
        self.init_plot()
        try:
            while True:
                data = self.callback_func()
                self.datas.append(data)
                #
                self.plot1.set_data(range(len(self.datas)), self.datas)
                #
                self.axs.relim()
                self.axs.autoscale_view()
                self.fig.canvas.flush_events()
                #
                time.sleep(0.05)
        except Exception as e:
            print(e)
        finally:
            plt.ioff()


def plot_ion_position_transmission(
    uuid: str, callback_func: Callable, HIST_LENGTH: int = 1000, optimize=False
):
    datas = deque(maxlen=HIST_LENGTH)  # [(x1,y1,T1), (x2,y2,T2), ...)]
    # >>> plot data <<<
    plt.ion()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # axs[0].set_xlabel("x(um)")
    # axs[0].set_ylabel("y(um)")
    axs[0].set(xlim=(-10, 10), ylim=(-10, 10), xlabel="y(um)", ylabel="x(um)")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("transmission")
    # ax.set_ylim(0, 1)
    fig_new_pt = axs[0].scatter([], [], marker="x", c="black", s=50)
    fig_position = axs[0].scatter([], [], c=np.array([]), cmap="jet", vmin=0, vmax=1)
    (fig_transmission,) = axs[1].plot([], [])
    # print(aaa)
    # >>> update data <<<

    def _optimize_wrapper(datas, callback_func, paras):
        # >>> update plot <<<
        if len(datas) > 0:
            xs, ys, es = zip(*datas)
            fig_position.set_offsets(np.column_stack((ys, xs)))
            fig_position.set_array(es)
            fig_position.set_alpha(
                np.linspace(0.2, 0.4, len(xs))
            )  # alpha increase with time
            #
            fig_transmission.set_data(range(len(es)), es)
            axs[1].relim()
            axs[1].autoscale_view()
        # highlight the new point
        fig_new_pt.set_offsets(np.column_stack((paras[1], paras[0])))
        x, y, T = callback_func(paras)
        # >>> update data <<<
        datas.append((x, y, T))
        #
        fig.canvas.flush_events()
        #
        return -T

    #
    try:
        if optimize:  # using scipy.optimize.minimize
            paras = [0, 0]
            paras = minimize(
                lambda para: _optimize_wrapper(datas, callback_func, para),
                x0=paras,
                # method="Nelder-Mead",
                method="Powell",
                # method="L-BFGS-B",
                bounds=[(-10, 10), (-10, 10)],
                options={"disp": True, "maxiter": 80, "ftol": 1e-10},
            )
            paras = paras.x
        else:  # 2D scanning
            X = np.linspace(-5, 5, 11)
            Y = np.linspace(-5, 5, 11)
            # snake scan
            max_T = 0
            bstPara = [0, 0]
            for i in range(len(X)):
                x = X[i]
                Yx = Y if i % 2 == 0 else Y[::-1]
                for y in Yx:
                    paras = [x, y]
                    minus_T = _optimize_wrapper(datas, callback_func, paras)
                    if minus_T < max_T:
                        max_T = minus_T
                        bstPara = paras
            paras = bstPara
            callback_func(paras)
        #
        plt.ioff()
        # >>> plot datas <<<
        plt.figure()
        xs, ys, es = zip(*datas)
        plt.scatter(
            ys, xs, c=es, cmap="jet", vmin=0, vmax=np.max(es), label="Scan on 2D grid"
        )
        xbst, ybst = paras
        plt.scatter(ybst, xbst, marker="x", c="black", s=50, label="Bst point")  # type: ignore
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.xlabel("y(um)")
        plt.ylabel("x(um)")
        plt.legend()
        plt.colorbar()
        dataName = getDataName(uuid)
        fileName = dataName + "_2D_align.png"
        print(fileName)
        plt.savefig(fileName, dpi=200, bbox_inches="tight")
        plt.show()

    except Exception as e:
        print(e)
    finally:
        res = input(
            "Accept the Final Position x={:.4f}(um), y={:.4f}(um) (y/n)?".format(*paras)  # type: ignore
        ).strip()
        if res == "n":
            callback_func([0, 0])
        else:
            pass
        #
        return
