import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.data_management import database

# import glob
import os
from pathlib import Path
import matplotlib.font_manager

matplotlib.rc("text", usetex=True)
matplotlib.rc("font", **{"family": "TeX Gyre DejaVu Math"})
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


def read_files_data(base_path):
    files_names = list(base_path.parent.glob(base_path.stem + "*"))
    files_names = [file_name for file_name in files_names if file_name.suffix == ".db"]
    print(files_names)

    n_tasks = [int(file_name.stem.split("_")[-1]) for file_name in files_names]
    print(n_tasks)

    res = dict()
    res["cpu_time"] = []
    res["n_tasks"] = []
    for n in range(len(n_tasks)):
        file = files_names[n]
        if not os.path.isfile(file):
            print(f"File {file} not found")
            continue

        file = os.path.splitext(file)[0]
        data_man = database(file)
        sim_data = data_man.read_dictionary("sim_data")
        res["cpu_time"].append(sim_data["cpu_time"])
        res["n_tasks"].append(n_tasks[n])

    res["cpu_time"] = np.array(res["cpu_time"])
    res["n_tasks"] = np.array(res["n_tasks"])
    sorted_ind = res["n_tasks"].argsort()
    res["cpu_time"][:] = res["cpu_time"][sorted_ind]
    res["n_tasks"][:] = res["n_tasks"][sorted_ind]

    return res


def get_plot_settings():
    markers = ["o", "x", "s", ">", ".", "<", ",", "1", "2", "3", "4", "v", "p", "*", "h", "H", "+", "^", "D", "d", "|", "_"]
    colors = ["k"] + [f"C{i}" for i in range(10)]
    colors[2], colors[3], colors[4], colors[8] = colors[3], colors[4], colors[8], colors[2]
    # colors[4], colors[2] = colors[2], colors[4]
    markerfacecolors = ["none" for _ in range(len(colors))]
    markerfacecolors[4] = colors[4]
    markersizes = [7.5 for _ in range(len(colors))]
    markeredgewidths = [1.2 for _ in range(len(colors))]
    figsize = (3, 2)
    return markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize


def print_results(results):
    for integrator, res in results.items():
        print(integrator)
        print(f'n_tasks = {res["n_tasks"]}')
        print(f'cpu_time = {res["cpu_time"]}')


def label_from_key(key):
    if key not in ["dt", "cpu_time", "rel_error_L2", "err_CV", "s_avg", "m_avg", "n_tasks"]:
        label = key.split("_", 1)[0]
        if label not in ["IMEXEXP", "err_CV"]:
            label = key.split("ES", 1)[0]
            if label == "exp_m":
                label = "em"
            label = label + ("RKC" if "RKC" in key else "RKW")
            label = label + (" P" if "prog" in key else "")
            safe_add = key.split("safe_add_")[1][0]
            if int(safe_add) > 0:
                label = label + " $s +\hspace{-3pt}= " + safe_add + "$"
        if label == "IMEXEXP":
            label = "IMEX-RL"
        return label
    else:
        if key == "dt":
            return "$\Delta t$"
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


def plot_scalability(results, legend_pos, output_path, save_plots_to_disk, show_plots, figure_title, with_legend):
    fs_label = 12
    fs_tick = 12
    fs_title = 12

    # in log log scale
    markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize = get_plot_settings()
    fig, ax = plt.subplots(figsize=(5, 3))
    i = 2
    ax.plot(
        results["exp_mES"]["n_tasks"],
        results["exp_mES"]["cpu_time"],
        label="emRKC",
        marker=markers[i],
        color=colors[i],
        linewidth=2,
        markerfacecolor=markerfacecolors[i],
        markeredgewidth=markeredgewidths[i],
        markersize=markersizes[i],
    )
    i = 0
    ax.plot(
        results["IMEXEXP"]["n_tasks"],
        results["IMEXEXP"]["cpu_time"],
        label="IMEX-RL",
        marker=markers[i],
        color=colors[i],
        linewidth=2,
        markerfacecolor=markerfacecolors[i],
        markeredgewidth=markeredgewidths[i],
        markersize=markersizes[i],
    )
    ax.plot(results["exp_mES"]["n_tasks"], 1600 / results["exp_mES"]["n_tasks"], label="Optimal", color="k", linewidth=2, linestyle="dashed")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(label_from_key("n_tasks"), fontsize=fs_label)
    ax.set_ylabel(label_from_key("cpu_time"), fontsize=fs_label)
    ax.tick_params(axis="x", labelsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.set_xticks(results["exp_mES"]["n_tasks"])
    ax.set_xticklabels(results["exp_mES"]["n_tasks"])
    if figure_title != "":
        ax.set_title(figure_title, fontsize=fs_title)
    # plt.tight_layout()
    if with_legend:
        ax.legend(loc=legend_pos)
    if save_plots_to_disk:
        fig.savefig(output_path, bbox_inches="tight", format="pdf")
    if show_plots:
        plt.show()


def import_and_plot_results(domain_name, ionic_model, pre_refinements, post_refinements, integrators, fibrosis, output_folder, save_plots_to_disk, show_plots, results_root, with_legend):
    results_subfolder = Path(domain_name) / Path("ref_" + str(pre_refinements)) / Path(ionic_model)
    folder = results_root / results_subfolder

    results = dict()

    for integrator in integrators:
        integrator_name = Path(integrator + "_post_ref_" + str(post_refinements) + "_n_tasks_")
        res = read_files_data(folder / integrator_name)
        if res != dict():
            results[integrator] = res

    if results == dict():
        return

    print_results(results)

    output_folder = output_folder
    if not output_folder.is_dir():
        os.makedirs(output_folder)

    output_file_name = domain_name + "_pre_ref_" + str(pre_refinements) + "_" + "post_ref_" + str(post_refinements) + "_" + ionic_model + ("_fibrosis" if fibrosis else "") + ".pdf"
    print(output_file_name)

    figure_title = ""

    plot_scalability(results, "upper right", output_folder / Path("scalability_" + output_file_name), save_plots_to_disk, show_plots, figure_title, with_legend)


def main():
    results_source = "daint"
    results_root_folder = "./Stabilized_integrators/"
    output_folder = Path.home() / "Dropbox/Applicazioni/Overleaf/Exponential Explicit Stabilized/images"

    save_plots_to_disk = False
    show_plots = True
    with_legend = True

    domain_name = "cuboid_3D"
    domain_name = "03_fastl_LA"
    pre_refinements = 2
    post_refinements = 0
    ionic_model = "CRN"
    fibrosis = False

    integrators = ["exp_mES", "IMEXEXP"]

    results_root = Path(results_root_folder + "/results_scalability" + ("_fibrosis" if fibrosis else "") + (("_" + results_source) if results_source != "" else ""))
    import_and_plot_results(domain_name, ionic_model, pre_refinements, post_refinements, integrators, fibrosis, output_folder, save_plots_to_disk, show_plots, results_root, with_legend)


if __name__ == "__main__":
    main()
