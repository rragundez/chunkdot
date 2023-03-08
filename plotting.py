import numpy as np
from matplotlib import pyplot as plt


def plot_memory(n_items, max_memory, matrix_memory):
    plt.figure(facecolor="white")
    plt.plot(n_items, [n / 1e9 for n in max_memory], color="cyan", label="Max memory")
    plt.plot(
        n_items, [n / 1e9 for n in matrix_memory], color="purple", label="Matrix memory"
    )
    plt.scatter(
        n_items,
        [8 * n**2 / 1e9 for n in n_items],
        color="green",
        marker="*",
        label="8 bytes * n^2",
    )
    plt.legend(loc="upper left")
    plt.ylabel("GB"), plt.xlabel("N items")
    plt.show()


def plot_time(n_items, execution_time):
    plt.figure(facecolor="white")
    plt.plot(n_items, execution_time, color="purple", label="Execution time")
    coeff_2, coeff_1, coeff_0 = np.polyfit(n_items, execution_time, 2)
    plt.scatter(
        n_items,
        [coeff_0 + coeff_1 * n + coeff_2 * n**2 for n in n_items],
        color="green",
        marker="*",
        label="Seconds",
    )
    plt.legend(loc="upper left")
    plt.show()
