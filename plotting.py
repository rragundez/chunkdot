import numpy as np
from matplotlib import pyplot as plt


def plot_memory(n_items, max_memory, matrix_memory):
    """Plot the memory consumption in GB vs the number of items."""
    plt.figure(facecolor="white")
    plt.plot(n_items, [n / 1e9 for n in max_memory], color="cyan", label="Max memory")
    plt.plot(n_items, [n / 1e9 for n in matrix_memory], color="purple", label="Matrix memory")
    plt.scatter(
        n_items,
        [8 * n**2 / 1e9 for n in n_items],
        color="green",
        marker="*",
        label="8 bytes * n^2",
    )
    plt.legend(loc="upper left")
    plt.ylabel("Memory in GB")
    plt.xlabel("Number of items")
    plt.show()


def plot_time(n_items, execution_time):
    """Plot the execution time vs the number of items."""
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
