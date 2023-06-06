import numpy as np
from stiffnessTensor import StiffnessTensor
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def plot_young_3d(stiffness_matrix: StiffnessTensor) -> None:
    """3D plotter for Young modulus"""

    n_points = 200

    theta_array = np.linspace(0, np.pi, n_points)
    phi_array = np.linspace(0, 2 * np.pi, 2 * n_points)

    data_x = np.zeros((n_points, 2*n_points))
    data_y = np.zeros((n_points, 2*n_points))
    data_z = np.zeros((n_points, 2*n_points))
    data_young = np.zeros((n_points, 2*n_points))

    for i in range(n_points):
        for j in range(2*n_points):
            young = stiffness_matrix.young((theta_array[i], phi_array[j]))
            data_young[i,j] = young
            z = young * np.cos(theta_array[i])
            x = young * np.sin(theta_array[i]) * np.cos(phi_array[j])
            y = young * np.sin(theta_array[i]) * np.sin(phi_array[j])
            data_x[i,j] = x
            data_y[i,j] = y
            data_z[i,j] = z

    young_average = np.average(data_young)
    young_min = np.min(data_young)
    young_max = np.max(data_young)

    figure = plt.figure()
    axes = plt.axes(projection='3d')

    norm = colors.Normalize(vmin=young_min, vmax=young_max, clip=False)

    surface = axes.plot_surface(data_x, data_y, data_z, norm=norm, cmap='viridis')

    scalarmap = cm.ScalarMappable(cmap='viridis', norm=norm)
    scalarmap.set_clim(young_min, young_max)

    cbar = plt.colorbar(scalarmap, orientation="horizontal", fraction=0.06, pad=-0.1, ticks=[young_min, young_average, young_max])
    cbar.ax.tick_params(labelsize='large')
    cbar.set_label(r'directional stiffness $E$ (MPa)', size=15, labelpad=20)

    axes.figure.axes[1].tick_params(axis="x", labelsize=20)
    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_young.png", transparent=True)
    plt.show()


def plot_linear_compressibility_3d(stiffness_matrix: StiffnessTensor) -> None:
    """3D plotter for linear compressibility modulus"""

    n_points = 200

    theta_array = np.linspace(0, np.pi, n_points)
    phi_array = np.linspace(0, 2 * np.pi, 2 * n_points)

    data_x_pos = np.zeros((len(theta_array), len(phi_array)))
    data_y_pos = np.zeros((len(theta_array), len(phi_array)))
    data_z_pos = np.zeros((len(theta_array), len(phi_array)))

    data_x_neg = np.zeros((len(theta_array), len(phi_array)))
    data_y_neg = np.zeros((len(theta_array), len(phi_array)))
    data_z_neg = np.zeros((len(theta_array), len(phi_array)))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):
            tmp_pos = max(0.0, stiffness_matrix.linear_compressibility((theta_array[index_theta], phi_array[index_phi])))
            tmp_neg = max(0.0, stiffness_matrix.linear_compressibility((theta_array[index_theta], phi_array[index_phi])))

            x_pos = tmp_pos * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y_pos = tmp_pos * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            z_pos = tmp_pos * np.cos(theta_array[index_theta])

            x_neg = tmp_neg * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y_neg = tmp_neg * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            z_neg = tmp_neg * np.cos(theta_array[index_theta])

            data_x_pos[index_theta][index_phi] = x_pos
            data_y_pos[index_theta][index_phi] = y_pos
            data_z_pos[index_theta][index_phi] = z_pos

            data_x_neg[index_theta][index_phi] = x_neg
            data_y_neg[index_theta][index_phi] = y_neg
            data_z_neg[index_theta][index_phi] = z_neg
