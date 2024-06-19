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
    files_names.sort()
    print(files_names)
    n_files = len(files_names)
    print(base_path)
    if n_files == 0:
        return False
    files_data = dict()
    files_data["error_L2_availabe"] = [False for _ in range(n_files)]
    files_data["error_L2"] = np.zeros(n_files)
    files_data["rel_error_L2"] = np.zeros(n_files)
    files_data["dt"] = np.zeros(n_files)
    files_data["cpu_time"] = np.zeros(n_files)
    files_data["CV"] = np.zeros(n_files)
    files_data["s_avg"] = np.zeros(n_files)
    files_data["m_avg"] = np.zeros(n_files)

    for n, file in enumerate(files_names):
        file = os.path.splitext(file)[0]
        data_man = database(file)
        # problem_params = data_man.read_dictionary("problem_params")
        int_params = data_man.read_dictionary("int_params")
        sim_data = data_man.read_dictionary("sim_data")
        sol_data = data_man.read_dictionary("sol_data")
        step_stats = data_man.read_dictionary("step_stats")
        files_data["error_L2_availabe"][n] = sol_data["error_L2_availabe"]
        files_data["error_L2"][n] = sol_data["error_L2"]
        files_data["rel_error_L2"][n] = sol_data["rel_error_L2"]
        files_data["dt"][n] = int_params["dt"]
        files_data["cpu_time"][n] = sim_data["cpu_time"]
        if "CV" in sol_data:
            files_data["CV"][n] = sol_data["CV"]
        if "s_avg" in step_stats:
            files_data["s_avg"][n] = step_stats["s_avg"]
            files_data["m_avg"][n] = step_stats["m_avg"]

    return files_data


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


def plot_results(results, ax1, ax2, logx, logy, location, output_file, save_plots_to_disk, show_plots, figure_title, with_legend):
    markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize = get_plot_settings()
    fig, ax = plt.subplots(figsize=figsize)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    for i, (key, result) in enumerate(results.items()):
        ax.plot(
            result[ax1],
            result[ax2],
            label=label_from_key(key),
            linewidth=2,
            marker=markers[i],
            color=colors[i],
            markerfacecolor=markerfacecolors[i],
            markeredgewidth=markeredgewidths[i],
            markersize=markersizes[i],
        )

    if ax1 == "dt":
        max_min = -1.0
        for i, (key, result) in enumerate(results.items()):
            dts = result[ax1]
            if not np.all(np.isnan(result[ax2])):
                max_min = max(max_min, np.min([res for res in result[ax2] if not np.isnan(res)]))

        dts = dts[5:]
        ax.plot(dts, 2.0 * dts * max_min / dts[-1], label="$\mathcal{O}(\Delta t)$", linewidth=2, marker=None, color="k", linestyle="dashed")

    fs_label = 12
    fs_tick = 12
    ax.set_xlabel(label_from_key(ax1), fontsize=fs_label, labelpad=-0.5)
    ax.set_ylabel(label_from_key(ax2), fontsize=fs_label, labelpad=-0.5)
    ax.tick_params(axis="x", labelsize=fs_tick, pad=1)
    ax.tick_params(axis="y", labelsize=fs_tick, pad=0)

    if figure_title != "":
        ax.set_title(figure_title)

    if with_legend:
        ax.legend(loc=location)

    if show_plots:
        plt.show()

    if save_plots_to_disk:
        fig.savefig(output_file, bbox_inches="tight", format="pdf")
        # export_legend(ax.get_legend())

    plt.close(fig)


def export_legend(legend, filename="legend.png"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def plot_stages(results, logx, logy, location, output_file, save_plots_to_disk, show_plots, figure_title, with_legend):
    markers, markerfacecolors, markersizes, markeredgewidths, colors, figsize = get_plot_settings()
    fig, (ax1, ax2) = plt.subplots(figsize=figsize, nrows=2, ncols=1)
    if logx:
        ax1.set_xscale("log")
        ax2.set_xscale("log")
    if logy:
        ax1.set_yscale("log")
        ax2.set_yscale("log")

    for i, (key, result) in enumerate(results.items()):
        if "IMEXEXP" in key:
            continue
        ax1.plot(
            result["dt"],
            result["s_avg"],
            label=label_from_key(key),
            linewidth=2,
            marker=markers[i],
            color=colors[i],
            markerfacecolor=markerfacecolors[i],
            markeredgewidth=markeredgewidths[i],
            markersize=markersizes[i],
        )
        ax2.plot(
            result["dt"],
            result["m_avg"],
            label=label_from_key(key),
            linewidth=2,
            marker=markers[i],
            color=colors[i],
            markerfacecolor=markerfacecolors[i],
            markeredgewidth=markeredgewidths[i],
            markersize=markersizes[i],
        )

    fs_label = 12
    fs_tick = 12
    ax1.set_ylabel(label_from_key("s_avg"), fontsize=fs_label, labelpad=0.0)
    # ax1.tick_params(axis="x", labelsize=fs_tick, pad=1)
    ax1.set_xticks([])
    ax1.tick_params(axis="y", labelsize=fs_tick, pad=0)
    ax2.set_xlabel(label_from_key("dt"), fontsize=fs_label, labelpad=0.0)
    ax2.set_ylabel(label_from_key("m_avg"), fontsize=fs_label, labelpad=0.0)
    ax2.tick_params(axis="x", labelsize=fs_tick, pad=0)
    ax2.tick_params(axis="y", labelsize=fs_tick, pad=0)
    ax1.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
    ax2.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))

    if with_legend:
        ax1.legend(loc=location)

    if figure_title != "":
        fig.suptitle(figure_title)

    # pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0, pos.width * 0.5, pos.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax1.legend(loc=location)
    if show_plots:
        plt.show()

    if save_plots_to_disk:
        fig.savefig(output_file, bbox_inches="tight")

    plt.close(fig)


def print_results(results):
    for integrator, res in results.items():
        print(integrator)
        print(f'dt = {res["dt"]}')
        print(f'rel_error_L2 = {res["rel_error_L2"]}')
        print(f'cpu_time = {res["cpu_time"]}')


def label_from_key(key):
    if key not in ["dt", "cpu_time", "rel_error_L2", "err_CV", "s_avg", "m_avg"]:
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


def import_and_plot_results(domain_name, ionic_model, refinements, integrators, es_classes, safe_adds, fibrosis, error_type, output_folder, save_plots_to_disk, show_plots, results_root, with_legend):
    results_subfolder = Path(domain_name) / Path("ref_" + str(refinements)) / Path(ionic_model)
    folder = results_root / results_subfolder

    results = dict()
    safe_adds_found = []

    if error_type == "CV":
        res = read_files_data(folder / Path("ref_sol"))
        CV_ref = res["CV"][0]

    if "IMEXEXP" in integrators:
        integrator_name = Path("IMEXEXP_splitting_exp_nonstiff")
        res = read_files_data(folder / integrator_name)
        if res is not False:
            results[str(integrator_name)] = res

    if "mES" in integrators:
        for es_class in es_classes:
            for safe_add in safe_adds:
                integrator_name = Path("mES_" + es_class + "1_safe_add_" + str(safe_add) + "_splitting_stiff_nonstiff")
                res = read_files_data(folder / integrator_name)
                if res is not False:
                    results[str(integrator_name)] = res
                    safe_adds_found.append(safe_add)

    if "exp_mES" in integrators:
        for es_class in es_classes:
            for safe_add in safe_adds:
                integrator_name = Path("exp_mES_" + es_class + "1_safe_add_" + str(safe_add) + "_splitting_exp_nonstiff")
                res = read_files_data(folder / integrator_name)
                if res is not False:
                    results[str(integrator_name)] = res
                    safe_adds_found.append(safe_add)

    if "exp_mES_prog" in integrators:
        for es_class in es_classes:
            for safe_add in safe_adds:
                integrator_name = Path("exp_mES_prog_" + es_class + "1_safe_add_" + str(safe_add) + "_splitting_exp_nonstiff")
                res = read_files_data(folder / integrator_name)
                if res is not False:
                    results[str(integrator_name)] = res
                    safe_adds_found.append(safe_add)

    if error_type == "CV":
        for res in results.values():
            res["err_CV"] = np.abs(res["CV"] - CV_ref) / CV_ref

    # max_cpu_time = 0.
    # for res in results.values():
    #     max_cpu_time = max(max_cpu_time,np.max(res['cpu_time']))
    # for res in results.values():
    #     res['cpu_time'] /= max_cpu_time

    print_results(results)

    if results == dict():
        return

    safe_adds_found = list(set(safe_adds_found))
    safe_adds_str = ""
    for safe_add in safe_adds_found:
        safe_adds_str = safe_adds_str + str(safe_add)

    output_folder = output_folder / Path(domain_name)

    if not output_folder.is_dir():
        os.makedirs(output_folder)

    output_file_name = "ref_" + str(refinements) + "_" + ionic_model + "_safe_adds_" + safe_adds_str + ("_fibrosis" if fibrosis else "") + ".pdf"
    # print(output_file_name)

    figure_title = ""

    if error_type == "CV":
        plot_results(results, "dt", "err_CV", True, True, "lower right", output_folder / Path("conv_" + output_file_name), save_plots_to_disk, show_plots, figure_title, with_legend)
        plot_results(results, "err_CV", "cpu_time", True, True, "lower left", output_folder / Path("eff_" + output_file_name), save_plots_to_disk, show_plots, figure_title, with_legend)
    else:
        plot_results(results, "dt", "rel_error_L2", True, True, "lower right", output_folder / Path("conv_" + output_file_name), save_plots_to_disk, show_plots, figure_title, with_legend)
        plot_results(results, "rel_error_L2", "cpu_time", True, True, "lower left", output_folder / Path("eff_" + output_file_name), save_plots_to_disk, show_plots, figure_title, with_legend)

    plot_stages(results, True, False, "upper left", output_folder / Path("stages_" + output_file_name), save_plots_to_disk, show_plots, figure_title, with_legend)


def main():
    results_source = ""
    results_root_folder = "./Stabilized_integrators_FeNICSx/results_paper/"
    output_folder = Path.home() / "Dropbox/Applicazioni/Overleaf/Exponential Explicit Stabilized/images"

    save_plots_to_disk = False
    show_plots = True
    with_legend = True

    domain_name = "cuboid_2D"
    # domain_name = "03_fastl_LA"
    refinements = 2
    ionic_model = "HH"
    fibrosis = False
    error_type = "L2"

    es_classes = ["RKC"]  # , "RKW"]
    integrators = ["exp_mES", "IMEXEXP", "mES"]  # , "exp_mES_prog"]
    safe_adds = [0]

    results_root = Path(results_root_folder + "/results_monodomain" + ("_fibrosis" if fibrosis else "") + (("_" + results_source) if results_source != "" else ""))
    import_and_plot_results(domain_name, ionic_model, refinements, integrators, es_classes, safe_adds, fibrosis, error_type, output_folder, save_plots_to_disk, show_plots, results_root, with_legend)


def print_all():
    results_source = ""
    results_root_folder = "./Stabilized_integrators_FeNICSx/results_paper/"
    output_folder = Path.home() / "Dropbox/Applicazioni/Overleaf/Exponential Explicit Stabilized/images"

    save_plots_to_disk = True
    show_plots = False
    with_legend = False

    domain_names = ["cuboid_2D"]  # ["03_fastl_LA"]  # ['cuboid_2D','cuboid_3D']
    refinementss = [0, 1, 2]
    ionic_models = ["TTP", "CRN"]
    fibrosiss = [False]
    error_type = "L2"

    integrators = ["IMEXEXP", "mES", "exp_mES"]
    es_classes = ["RKC"]  # ,'RKW']
    safe_adds = [0]

    for domain_name in domain_names:
        for refinements in refinementss:
            for ionic_model in ionic_models:
                for fibrosis in fibrosiss:
                    results_root = Path(results_root_folder + "/results_monodomain" + ("_fibrosis" if fibrosis else "") + (("_" + results_source) if results_source != "" else ""))
                    import_and_plot_results(
                        domain_name, ionic_model, refinements, integrators, es_classes, safe_adds, fibrosis, error_type, output_folder, save_plots_to_disk, show_plots, results_root, with_legend
                    )


if __name__ == "__main__":
    main()
    # print_all()
