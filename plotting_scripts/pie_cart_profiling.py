import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager
from pathlib import Path
import pstats

matplotlib.rc("text", usetex=True)
matplotlib.rc("font", **{"family": "TeX Gyre DejaVu Math"})
plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")


def plot_pie_chart(times, name, show_plots, save_plots):
    times["other"] = times["total"] - np.sum(np.array([t for key, t in times.items() if key != "total"]))
    total = times["total"]
    del times["total"]

    def func(pct, allvals):
        absolute = pct / 100.0 * np.sum(allvals)
        if pct >= 5.0:
            return f"{absolute:.1f} s"  # \n   {pct:.1f}\%"
        else:
            return ""

    data = list(times.values())
    labels = list(times.keys())
    data = [data[-1]] + data[0:-1]
    labels = [labels[-1]] + ["$" + lab + "$" for lab in labels[0:-1]]

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(data, labels=labels, autopct=lambda pct: func(pct, data), textprops=dict(color="k", size=14))

    plt.setp(autotexts, size=10, weight="bold", bbox=dict(facecolor="white", alpha=1.0, edgecolor="white"))

    ax.set_title(f"{total:.1f} s", loc="center", y=0.45, bbox=dict(facecolor="white", alpha=1.0, edgecolor="black"))

    if show_plots:
        plt.show()

    if save_plots:
        fig.savefig(name, bbox_inches="tight", format="pdf")


def read_prof(input_folder, model, method):
    folder = input_folder / Path(model)
    files_names = list(folder.glob(method + "*"))
    files_names = [file_name for file_name in files_names if file_name.suffix == ".prof"]
    files_names.sort()
    n_files = len(files_names)
    times = dict()
    functions_map = functions(method)
    for fun in functions_map:
        times[fun[0]] = 0.0
    for file_name in files_names:
        p = pstats.Stats(str(file_name))
        p.strip_dirs()
        sp = p.get_stats_profile()
        for fun in functions_map:
            times[fun[0]] += sp.func_profiles[fun[1]].cumtime

    for fun in functions_map:
        times[fun[0]] = times[fun[0]] / n_files

    return times


def functions(method):
    if method == "IMEXEXP":
        functions_map = [["total", "step"], ["F", "solve_system"], ["S", "eval_f_nonstiff"], ["E", "eval_phi_f_exp"]]
    if method == "mES":
        functions_map = [["total", "step"], ["F", "eval_f_stiff"], ["S", "eval_f_nonstiff"]]
    if method == "exp_mES":
        functions_map = [["total", "step"], ["F", "eval_f_stiff"], ["S", "eval_f_nonstiff"], ["E", "eval_phi_f_exp"]]

    return functions_map


def test_one_plot():
    output_folder = Path.home() / "Dropbox/Applicazioni/Overleaf/Exponential Explicit Stabilized/images/profiling/"
    root_folder = Path.home() / "Dropbox/Ricerca/Codes/Research_Codes/pySDC_and_Stabilized_in_FeNICSx/Stabilized_integrators_FeNICSx/"
    input_folder = root_folder / Path("results_monodomain_profiling_daint/03_fastl_LA/ref_2/")
    models = ["TTP"]  # , "CRN", "TTP"]
    methods = ["exp_mES"]  # , "mES", "IMEXEXP"]
    show_plots = True
    save_plots = False

    prof_res = dict()
    for method in methods:
        for model in models:
            prof_res[model + "_" + method] = read_prof(input_folder, model, method)

    for name, times in prof_res.items():
        plot_pie_chart(times, output_folder / Path(name).with_suffix(".pdf"), show_plots, save_plots)


def main():
    output_folder = Path.home() / "Dropbox/Applicazioni/Overleaf/Exponential Explicit Stabilized/images/profiling/"
    root_folder = Path.home() / "Dropbox/Ricerca/Codes/Research_Codes/pySDC_and_Stabilized_in_FeNICSx/Stabilized_integrators_FeNICSx/"
    input_folder = root_folder / Path("results_monodomain_profiling_daint/03_fastl_LA/ref_2/")
    models = ["HH", "CRN", "TTP"]
    methods = ["exp_mES", "IMEXEXP"]
    show_plots = False
    save_plots = True

    prof_res = dict()
    for method in methods:
        for model in models:
            prof_res[model + "_" + method] = read_prof(input_folder, model, method)

    for name, times in prof_res.items():
        plot_pie_chart(times, output_folder / Path(name).with_suffix(".pdf"), show_plots, save_plots)


if __name__ == "__main__":
    test_one_plot()
    # main()
