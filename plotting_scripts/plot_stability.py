import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.data_management import database

# import glob
import os
from pathlib import Path
import matplotlib.font_manager

# matplotlib.rc("text", usetex=True)
# matplotlib.rc("font", **{"family": "TeX Gyre DejaVu Math"})
# plt.rc("text", usetex=True)
# plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

matplotlib.rc("font", **{"family": "TeX Gyre DejaVu Math"})
plt.rc("text", usetex=True)


def get_plot_settings():
    markers = [
        "o",
        "x",
        "s",
        ">",
        ".",
        "<",
        ",",
        "1",
        "2",
        "3",
        "4",
        "v",
        "p",
        "*",
        "h",
        "H",
        "+",
        "^",
        "D",
        "d",
        "|",
        "_",
    ]
    colors = ["k"] + [f"C{i}" for i in range(10)]
    colors[2], colors[3], colors[4], colors[8] = colors[3], colors[4], colors[8], colors[2]
    # colors[4], colors[2] = colors[2], colors[4]
    markerfacecolors = ["none" for _ in range(len(colors))]
    markerfacecolors[4] = colors[4]
    markersizes = [7.5 for _ in range(len(colors))]
    markeredgewidths = [1.2 for _ in range(len(colors))]
    figsize = (3, 2)
    return markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize


def label_from_key(key):
    if key == "dt":
        return "$\Delta t$"
    elif key == "dx":
        return "$\Delta x$"
    elif key == "V_rel":
        return "$\Vert\mathbf{V_N}\Vert_{L^2(\Omega)}/\Vert\mathbf{V^*}\Vert_{L^2(\Omega)}$"
    elif key == "max_dt":
        return "$\Delta t_{\max{}}$"
    elif key == "cpu_time":
        return "CPU time"
    elif key == "rel_error_L2":
        return "$L^2$-norm rel. err."
    elif key == "err_CV":
        return "CV rel. err."
    elif key == "s_avg":
        return "s"
    elif key == "m_avg":
        return "m"
    elif key == "n_tasks":
        return "Number of Processes"


def main():
    output_folder = (
        Path.home()
        / "Dropbox/Ricerca/Articoli/Explicit stabilized methods/Articolo 9 - emRKC for Monodomain - Sottomesso/2nd Submission/images/"
    )

    save_plots_to_disk = True
    show_plots = True
    with_legend = True

    domain_name = "cuboid_1D"
    ionic_model = "TTP"

    output_file_name = domain_name + "_norm_V_VS_dt.pdf"

    fs_label = 12
    fs_tick = 12
    fs_title = 12
    legend_pos = "upper left"

    p = 3
    n_elems = 5 * 2**p
    dx = 1.0 / n_elems

    print("dx = ", dx)

    results = dict()
    results["exp_mES"] = dict()
    results["IMEXEXP"] = dict()
    results["exp_mES"]["dt"] = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.5, 6.4]
    results["exp_mES"]["s"] = [1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 5]
    results["exp_mES"]["m"] = [1, 2, 3, 3, 5, 6, 9, 6, 9, 8, 9, 10, 10]
    results["exp_mES"]["V_rel"] = [
        32.05226994263117,
        32.05216702017461,
        32.05196624796677,
        32.051618517919565,
        32.051196824384775,
        32.05130052672393,
        32.055499245189445,
        32.07337549621755,
        32.09498356223536,
        32.163019254369296,
        32.30107956898067,
        32.943376699849864,
        1000000,
    ]
    # results["IMEXEXP"]["dt"] = [0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    # results["IMEXEXP"]["V_rel"] = [32.05233127166095, 32.05227056361082, 32.0521579515064, 32.05196944567743, 32.05174995225189, 32.05197614896442, 32.05494494663634,32.06702191517687,32.09226474885607, 32.151909259896115,32.28633279172595,32.571290979310085]

    V_ref = 32.05226994263117
    results["exp_mES"]["V_rel"] = np.array(results["exp_mES"]["V_rel"]) / V_ref
    # results["IMEXEXP"]["V_rel"] = np.array(results["IMEXEXP"]["V_rel"]) / V_ref

    # in log log scale
    markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize = get_plot_settings()
    fig, ax = plt.subplots(figsize=figsize)
    i = 2
    ax.plot(
        results["exp_mES"]["dt"],
        results["exp_mES"]["V_rel"],
        label="emRKC",
        marker=markers[i],
        color=colors[i],
        linewidth=2,
        markerfacecolor=markerfacecolors[i],
        markeredgewidth=markeredgewidths[i],
        markersize=markersizes[i],
    )
    # i = 0
    # ax.plot(
    #     results["EXEXEXP"]["dx"],
    #     results["EXEXEXP"]["max_dt"],
    #     label="EXEX-RL",
    #     marker=markers[i],
    #     color=colors[i],
    #     linewidth=2,
    #     markerfacecolor=markerfacecolors[i],
    #     markeredgewidth=markeredgewidths[i],
    #     markersize=markersizes[i],
    # )
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xlabel(label_from_key("dt"), fontsize=fs_label)
    ax.set_ylabel(label_from_key("V_rel"), fontsize=fs_label)
    ax.tick_params(axis="x", labelsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)
    # ax.set_xticks([0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.8, 3.2, 6.4])
    # ax.set_xticklabels(results["exp_mES"]["dt"][::2])
    ax.set_yticks([0.1, 1, 10])
    ax.set_ylim([0.1, 10])

    # plt.tight_layout()
    if with_legend:
        ax.legend(loc=legend_pos)
    if save_plots_to_disk:
        fig.savefig(output_folder / Path(output_file_name), bbox_inches="tight", format="pdf")
    if show_plots:
        plt.show()

    # ----------------------------------------

    output_file_name = domain_name + "_sm_VS_dt.pdf"

    fig, ax = plt.subplots(figsize=figsize)
    i = 1
    ax.plot(
        results["exp_mES"]["dt"],
        results["exp_mES"]["s"],
        label="s",
        marker=markers[i],
        color=colors[i],
        linewidth=2,
        markerfacecolor=markerfacecolors[i],
        markeredgewidth=markeredgewidths[i],
        markersize=markersizes[i],
    )
    i = 2
    ax.plot(
        results["exp_mES"]["dt"],
        results["exp_mES"]["m"],
        label="m",
        marker=markers[i],
        color=colors[i],
        linewidth=2,
        markerfacecolor=markerfacecolors[i],
        markeredgewidth=markeredgewidths[i],
        markersize=markersizes[i],
    )
    ax.set_xscale("log", base=10)
    # ax.set_yscale("log", base=10)
    ax.set_xlabel(label_from_key("dt"), fontsize=fs_label)
    ax.set_ylabel("stages", fontsize=fs_label)
    ax.tick_params(axis="x", labelsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.set_yticks([1, 5, 10])
    # ax.set_xticklabels(results["exp_mES"]["dx"])
    # plt.tight_layout()
    if with_legend:
        ax.legend(loc=legend_pos)
    if save_plots_to_disk:
        fig.savefig(output_folder / Path(output_file_name), bbox_inches="tight", format="pdf")
    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
