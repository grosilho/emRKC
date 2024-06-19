import numpy as np
import matplotlib.pyplot as plt


def read_data(file_name):
    t_val = np.load(file_name + "_t.npy")
    u_val = np.load(file_name + "_u.npy")
    pts = np.load(file_name + "_p.npy")

    return t_val, u_val, pts


def compute_activation_times(t_val, u_val):
    n_pts = u_val.shape[0]
    v_th = -20
    t_th = np.zeros(n_pts)
    for i in range(n_pts):
        i_th = np.argmax(u_val[i, :] > v_th)
        if i_th == 0:
            t_th[i] = np.inf
        else:
            t_th[i] = t_val[i_th - 1] + (t_val[i_th] - t_val[i_th - 1]) * (v_th - u_val[i, i_th - 1]) / (
                u_val[i, i_th] - u_val[i, i_th - 1]
            )
    return t_th


def compute_CV(t_th, pts):
    n_pts = pts.shape[0]
    CV = 0.0
    for i in range(n_pts - 1):
        p1 = pts[i, :]
        p2 = pts[i + 1, :]
        dp = p2 - p1
        dt = t_th[i + 1] - t_th[i]
        CV += np.linalg.norm(dp) / dt
    CV /= n_pts - 1

    return CV


def my_plotter(data1, data2, ax, plot_options):
    """
    A helper function to make a graph.
    """
    out = ax.plot(data1, data2, **plot_options)
    ax.set_xlabel(r"$\Vert p \Vert$ [mm]", fontsize=16, color="black")
    ax.set_ylabel("$t$ [ms]", fontsize=16, color="black")
    ax.set_title("Activation times", fontsize=16)
    ax.tick_params(axis="x", labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0.0, 65)
    ax.legend(loc="upper left", fontsize=16)

    return out


def read_and_plot_data(filename, integrator, dt, es_class, ax, plot_options):
    folder = "./results/"
    file = filename + "_" + integrator
    if integrator in ["ES", "mES", "exp_mES"]:
        file += "_" + es_class
    file += "_dt_" + str(dt).replace(".", "p")

    print(f"Results for integrator = {integrator} and dt = {dt}")
    t_val, u_val, pts = read_data(folder + file)
    t_th = compute_activation_times(t_val, u_val)
    CV = compute_CV(t_th, pts)
    print(f"CV = {CV} m/s")

    pts_norms = np.linalg.norm(pts, axis=1)
    integrator_name = integrator + " " + es_class if integrator != "IMEXEXP" else "IMEX+RL"
    plot_options["label"] = integrator_name + r"$, \Delta t = $" + str(dt)
    my_plotter(pts_norms, t_th, ax, plot_options)
    print(f"Activation time in last point: {t_th[-1]}")


if __name__ == "__main__":
    # Results for monodomain_2d_refsol_n_elems_200_dt_0p001_TT_mES
    # CV = 0.4806660107652706
    # Activation time in last point: 42.13952652055738

    filename = "monodomain"
    es_class = "RKC1"
    integrator = "exp_mES"
    integrator = "IMEXEXP"
    # integrator = 'mES'
    dt1 = 0.01
    dt2 = 0.05

    lw = 3

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    read_and_plot_data(
        filename,
        integrator,
        dt=dt1,
        es_class=es_class,
        ax=ax,
        plot_options={"marker": "d", "linewidth": lw, "linestyle": "-"},
    )
    read_and_plot_data(
        filename,
        integrator,
        dt=dt2,
        es_class=es_class,
        ax=ax,
        plot_options={"marker": "d", "linewidth": lw, "linestyle": "-"},
    )
    ax.plot([21.4], [43.0], c="black", marker="o")
    plt.subplots_adjust(bottom=0.17, top=0.9, left=0.13, right=0.95)
    plt.show()
