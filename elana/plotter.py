import numpy as np
from stiffnessTensor import StiffnessTensor
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import cm, colors


def _symmetrical_colormap(cmap_settings, new_name=None):
    ''' This function take a colormap and create a new one, as the concatenation of itself by a symmetrical fold.
    '''
    # get the colormap
    cmap = plt.cm.get_cmap(*cmap_settings)
    if not new_name:
        new_name = "sym_" + cmap_settings[0]  # ex: 'sym_Blues'

    # this defines the roughness of the colormap, 128 fine
    n = 128

    # get the list of color from colormap
    colors_r = cmap(np.linspace(0, 1, n))  # take the standard colormap # 'right-part'
    colors_l = colors_r[::-1]  # take the first list of color and flip the order # "left-part"

    # combine them and build a new colormap
    new_colors = np.vstack((colors_l, colors_r))
    my_map = colors.LinearSegmentedColormap.from_list(new_name, new_colors)

    return my_map

def plot_young_3d(stiffness_matrix: StiffnessTensor) -> None:
    """3D plotter for Young modulus"""

    n_points = 200

    theta_array = np.linspace(0, np.pi, n_points)
    phi_array = np.linspace(0, 2 * np.pi, 2 * n_points)

    data_x = np.zeros((n_points, 2*n_points))
    data_y = np.zeros((n_points, 2*n_points))
    data_z = np.zeros((n_points, 2*n_points))
    data_young = np.zeros((n_points, 2*n_points))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):
            young = stiffness_matrix.young((theta_array[index_theta], phi_array[index_phi]))
            data_young[index_theta,index_phi] = young
            z = young * np.cos(theta_array[index_theta])
            x = young * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y = young * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            data_x[index_theta,index_phi] = x
            data_y[index_theta,index_phi] = y
            data_z[index_theta,index_phi] = z

    young_average = np.average(data_young)
    young_min = np.min(data_young)
    young_max = np.max(data_young)

    plt.figure()
    axes = plt.axes(projection='3d')

    norm = colors.Normalize(vmin=young_min, vmax=young_max, clip=False)

    axes.plot_surface(data_x, data_y, data_z, norm=norm, cmap='viridis')

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

    data_linear_compressibility_pos = np.zeros((n_points, 2*n_points))
    data_linear_compressibility_neg = np.zeros((n_points, 2 * n_points))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):
            linear_compressibility_pos = max(0.0, stiffness_matrix.linear_compressibility((theta_array[index_theta], phi_array[index_phi])))
            linear_compressibility_neg = max(0.0, -stiffness_matrix.linear_compressibility((theta_array[index_theta], phi_array[index_phi])))

            data_linear_compressibility_pos[index_theta, index_phi] = linear_compressibility_pos
            data_linear_compressibility_neg[index_theta, index_phi] = linear_compressibility_neg

            x_pos = linear_compressibility_pos * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y_pos = linear_compressibility_pos * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            z_pos = linear_compressibility_pos * np.cos(theta_array[index_theta])

            x_neg = linear_compressibility_neg * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y_neg = linear_compressibility_neg * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            z_neg = linear_compressibility_neg * np.cos(theta_array[index_theta])

            data_x_pos[index_theta][index_phi] = x_pos
            data_y_pos[index_theta][index_phi] = y_pos
            data_z_pos[index_theta][index_phi] = z_pos

            data_x_neg[index_theta][index_phi] = x_neg
            data_y_neg[index_theta][index_phi] = y_neg
            data_z_neg[index_theta][index_phi] = z_neg

    linear_compressibility_max = max(data_linear_compressibility_pos)
    linear_compressibility_min = min(data_linear_compressibility_neg)

    plt.figure()
    axes = plt.axes(projection='3d')

    norm_pos = colors.Normalize(vmin=0.0, vmax=linear_compressibility_max, clip=False)
    norm_neg = colors.Normalize(vmin=linear_compressibility_min, vmax=0.0, clip=False)
    norm = colors.Normalize(vmin=linear_compressibility_min, vmax=linear_compressibility_max, clip=False)

    axes.plot_surface(data_x_pos, data_y_pos, data_z_pos, norm=norm_pos, cmap='viridis')
    axes.plot_surface(data_x_neg, data_y_neg, data_z_neg, norm=norm_neg, cmap='viridis')

    scalarmap = cm.ScalarMappable(cmap='viridis', norm=norm)
    scalarmap.set_clim(linear_compressibility_min, linear_compressibility_max)

    cbar = plt.colorbar(scalarmap, orientation="horizontal", fraction=0.06, pad=-0.1, ticks=[linear_compressibility_min, 0.0, linear_compressibility_max])
    cbar.ax.tick_params(labelsize='large')
    cbar.set_label(r'directional stiffness $E$ (MPa)', size=15, labelpad=20)

    axes.figure.axes[1].tick_params(axis="x", labelsize=20)
    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_linear_compressibility.png", transparent=True)
    plt.show()