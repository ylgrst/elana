import numpy as np
from elana.stiffness_tensor import StiffnessTensor
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


_symmetrical_viridis = _symmetrical_colormap(('viridis', None))
_symmetrical_blues = _symmetrical_colormap(('Blues', None))
_symmetrical_greens = _symmetrical_colormap(('Greens', None))
_symmetrical_reds = _symmetrical_colormap(('Reds', None))


def plot_young_2d(stiffness_matrix: StiffnessTensor) -> None:
    """2D plotter for Young modulus"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    young_xy = list(map(lambda x: stiffness_matrix.young((np.pi / 2, x)), theta_array))
    young_xz = list(map(lambda x: stiffness_matrix.young((x, 0.0)), theta_array))
    young_yz = list(map(lambda x: stiffness_matrix.young((x, np.pi)), theta_array))

    data_x_xy = young_xy * np.cos(theta_array)
    data_y_xy = young_xy * np.sin(theta_array)

    data_x_xz = young_xz * np.sin(theta_array)
    data_y_xz = young_xz * np.cos(theta_array)

    data_x_yz = young_yz * np.sin(theta_array)
    data_y_yz = young_yz * np.sin(theta_array)

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3)
    ax_xy.plot(data_x_xy, data_y_xy, 'g-')
    ax_xy.set_title("Young modulus on (xy) plane")
    ax_xz.plot(data_x_xz, data_y_xz, 'g-')
    ax_xz.set_title("Young modulus on (xz) plane")
    ax_yz.plot(data_x_yz, data_y_yz, 'g-')
    ax_yz.set_title("Young modulus on (yz) plane")

    plt.savefig("planar_young.png", transparent=True)
    plt.show()


def plot_young_3d(stiffness_matrix: StiffnessTensor) -> None:
    """3D plotter for Young modulus"""

    n_points = 200

    theta_array = np.linspace(0.0, np.pi, n_points)
    phi_array = np.linspace(0.0, 2 * np.pi, 2 * n_points)

    data_x = np.zeros((n_points, 2 * n_points))
    data_y = np.zeros((n_points, 2 * n_points))
    data_z = np.zeros((n_points, 2 * n_points))
    data_young = np.zeros((n_points, 2 * n_points))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):
            young = stiffness_matrix.young((theta_array[index_theta], phi_array[index_phi]))
            data_young[index_theta, index_phi] = young
            z = young * np.cos(theta_array[index_theta])
            x = young * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y = young * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            data_x[index_theta, index_phi] = x
            data_y[index_theta, index_phi] = y
            data_z[index_theta, index_phi] = z

    young_average = np.average(data_young)
    young_min = np.min(data_young)
    young_max = np.max(data_young)

    plt.figure()
    axes = plt.axes(projection='3d')

    norm = colors.Normalize(vmin=young_min, vmax=young_max, clip=False)

    axes.plot_surface(data_x, data_y, data_z, norm=norm, cmap='viridis')

    scalarmap = cm.ScalarMappable(cmap='viridis', norm=norm)
    scalarmap.set_clim(young_min, young_max)

    cbar = plt.colorbar(scalarmap, orientation="horizontal", fraction=0.06, pad=-0.1,
                        ticks=[young_min, young_average, young_max])
    cbar.ax.tick_params(labelsize='large')
    cbar.set_label(r'directional stiffness $E$ (MPa)', size=15, labelpad=20)

    axes.figure.axes[1].tick_params(axis="x", labelsize=20)
    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_young.png", transparent=True)
    plt.show()


def plot_linear_compressibility_2d(stiffness_matrix: StiffnessTensor) -> None:
    """2D plotter for linear compressibility modulus"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    linear_compressibility_pos_xy = list(map(lambda x: max(0.0, stiffness_matrix.linear_compressibility((np.pi/2.0, x))), theta_array))
    linear_compressibility_pos_xz = list(
        map(lambda x: max(0.0, stiffness_matrix.linear_compressibility((x, 0.0))), theta_array))
    linear_compressibility_pos_yz = list(
        map(lambda x: max(0.0, stiffness_matrix.linear_compressibility((x, np.pi/2.0))), theta_array))

    data_x_xy_pos = linear_compressibility_pos_xy*np.cos(theta_array)
    data_y_xy_pos = linear_compressibility_pos_xy*np.sin(theta_array)
    data_x_xz_pos = linear_compressibility_pos_xz*np.sin(theta_array)
    data_y_xz_pos = linear_compressibility_pos_xz*np.cos(theta_array)
    data_x_yz_pos = linear_compressibility_pos_yz*np.sin(theta_array)
    data_y_yz_pos = linear_compressibility_pos_yz*np.cos(theta_array)

    linear_compressibility_neg_xy = list(map(lambda x: max(0.0, -stiffness_matrix.linear_compressibility((np.pi/2.0, x))), theta_array))
    linear_compressibility_neg_xz = list(
        map(lambda x: max(0.0, -stiffness_matrix.linear_compressibility((x, 0.0))), theta_array))
    linear_compressibility_neg_yz = list(
        map(lambda x: max(0.0, -stiffness_matrix.linear_compressibility((x, np.pi/2.0))), theta_array))

    data_x_xy_neg = linear_compressibility_neg_xy*np.cos(theta_array)
    data_y_xy_neg = linear_compressibility_neg_xy*np.sin(theta_array)
    data_x_xz_neg = linear_compressibility_neg_xz*np.sin(theta_array)
    data_y_xz_neg = linear_compressibility_neg_xz*np.cos(theta_array)
    data_x_yz_neg = linear_compressibility_neg_yz*np.sin(theta_array)
    data_y_yz_neg = linear_compressibility_neg_yz*np.cos(theta_array)

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3)
    ax_xy.plot(data_x_xy_pos, data_y_xy_pos, 'g-')
    ax_xy.plot(data_x_xy_neg, data_y_xy_neg, 'r-')
    ax_xy.set_title("Linear compressibility on (xy) plane")
    ax_xz.plot(data_x_xz_pos, data_y_xz_pos, 'g-')
    ax_xz.plot(data_x_xz_neg, data_y_xz_neg, 'r-')
    ax_xz.set_title("Linear compressibility on (xz) plane")
    ax_yz.plot(data_x_yz_pos, data_y_yz_pos, 'g-')
    ax_yz.plot(data_x_yz_neg, data_y_yz_neg, 'r-')
    ax_yz.set_title("Linear compressibility on (yz) plane")

    plt.savefig("planar_linear_compressibility.png", transparent=True)
    plt.show()




def plot_linear_compressibility_3d(stiffness_matrix: StiffnessTensor) -> None:
    """3D plotter for linear compressibility modulus"""

    n_points = 200

    theta_array = np.linspace(0.0, np.pi, n_points)
    phi_array = np.linspace(0.0, 2 * np.pi, 2 * n_points)

    data_x_pos = np.zeros((len(theta_array), len(phi_array)))
    data_y_pos = np.zeros((len(theta_array), len(phi_array)))
    data_z_pos = np.zeros((len(theta_array), len(phi_array)))

    data_x_neg = np.zeros((len(theta_array), len(phi_array)))
    data_y_neg = np.zeros((len(theta_array), len(phi_array)))
    data_z_neg = np.zeros((len(theta_array), len(phi_array)))

    data_linear_compressibility_pos = np.zeros((n_points, 2 * n_points))
    data_linear_compressibility_neg = np.zeros((n_points, 2 * n_points))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):
            linear_compressibility_pos = max(0.0, stiffness_matrix.linear_compressibility(
                (theta_array[index_theta], phi_array[index_phi])))
            linear_compressibility_neg = max(0.0, -stiffness_matrix.linear_compressibility(
                (theta_array[index_theta], phi_array[index_phi])))

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

    scalarmap = cm.ScalarMappable(cmap=_symmetrical_viridis, norm=norm)
    scalarmap.set_clim(linear_compressibility_min, linear_compressibility_max)

    cbar = plt.colorbar(scalarmap, orientation="horizontal", fraction=0.06, pad=-0.1,
                        ticks=[linear_compressibility_min, 0.0, linear_compressibility_max])
    cbar.ax.tick_params(labelsize='large')
    cbar.set_label(r'directional linear compressibility $LC$ (MPa $^{-1}$)', size=15, labelpad=20)

    axes.figure.axes[1].tick_params(axis="x", labelsize=20)
    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_linear_compressibility.png", transparent=True)
    plt.show()


def plot_shear_modulus_2d(stiffness_matrix: StiffnessTensor) -> None:
    """2D plotter for shear modulus"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    shear_xy = list(map(lambda x: stiffness_matrix.shear_2d((np.pi/2.0, x)), theta_array))
    shear_xz = list(map(lambda x: stiffness_matrix.shear_2d((x, 0.0)), theta_array))
    shear_yz = list(map(lambda x: stiffness_matrix.shear_2d((x, np.pi/2.0)), theta_array))

    data_x_xy_min = np.array([shear[0] * np.cos(theta) for shear, theta in zip(shear_xy, theta_array)])
    data_y_xy_min = np.array([shear[0] * np.sin(theta) for shear, theta in zip(shear_xy, theta_array)])
    data_x_xy_max = np.array([shear[1] * np.cos(theta) for shear, theta in zip(shear_xy, theta_array)])
    data_y_xy_max = np.array([shear[1] * np.sin(theta) for shear, theta in zip(shear_xy, theta_array)])

    data_x_xz_min = np.array([shear[0] * np.sin(theta) for shear, theta in zip(shear_xz, theta_array)])
    data_y_xz_min = np.array([shear[0] * np.cos(theta) for shear, theta in zip(shear_xz, theta_array)])
    data_x_xz_max = np.array([shear[1] * np.sin(theta) for shear, theta in zip(shear_xz, theta_array)])
    data_y_xz_max = np.array([shear[1] * np.cos(theta) for shear, theta in zip(shear_xz, theta_array)])

    data_x_yz_min = np.array([shear[0] * np.sin(theta) for shear, theta in zip(shear_yz, theta_array)])
    data_y_yz_min = np.array([shear[0] * np.cos(theta) for shear, theta in zip(shear_yz, theta_array)])
    data_x_yz_max = np.array([shear[1] * np.sin(theta) for shear, theta in zip(shear_yz, theta_array)])
    data_y_yz_max = np.array([shear[1] * np.cos(theta) for shear, theta in zip(shear_yz, theta_array)])

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3)
    ax_xy.plot(data_x_xy_max, data_y_xy_max, 'b-')
    ax_xy.plot(data_x_xy_min, data_y_xy_min, 'g-')
    ax_xy.set_title("Shear modulus on (xy) plane")
    ax_xz.plot(data_x_xz_max, data_y_xz_max, 'b-')
    ax_xz.plot(data_x_xz_min, data_y_xz_min, 'g-')
    ax_xz.set_title("Shear modulus on (xz) plane")
    ax_yz.plot(data_x_yz_max, data_y_yz_max, 'b-')
    ax_yz.plot(data_x_yz_min, data_y_yz_min, 'g-')
    ax_yz.set_title("Shear modulus on (yz) plane")

    plt.savefig("planar_shear_modulus.png", transparent=True)
    plt.show()


def plot_shear_modulus_3d(stiffness_matrix: StiffnessTensor) -> None:
    """3D plotter for shear modulus"""

    n_points = 200

    theta_array = np.linspace(0.0, np.pi, n_points)
    phi_array = np.linspace(0.0, np.pi, n_points)
    phi_plus_pi_array = [phi_array[i] + np.pi for i in range(1, len(phi_array))]
    phi_array = np.append(phi_array, phi_plus_pi_array)

    data_x_shear_min = np.zeros((len(theta_array), len(phi_array)))
    data_y_shear_min = np.zeros((len(theta_array), len(phi_array)))
    data_z_shear_min = np.zeros((len(theta_array), len(phi_array)))
    data_x_shear_max = np.zeros((len(theta_array), len(phi_array)))
    data_y_shear_max = np.zeros((len(theta_array), len(phi_array)))
    data_z_shear_max = np.zeros((len(theta_array), len(phi_array)))

    data_shear_max = np.zeros((n_points, 2 * n_points))
    data_shear_min = np.zeros((n_points, 2 * n_points))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):
            shear = stiffness_matrix.shear_3d((theta_array[index_theta], phi_array[index_phi]))
            data_shear_min[index_theta, index_phi] = shear[0]
            data_shear_max[index_theta, index_phi] = shear[1]
            z = np.cos(theta_array[index_theta])
            x = np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y = np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])

            shear_min = shear[0]
            z_min = shear_min * z
            x_min = shear_min * x
            y_min = shear_min * y
            data_x_shear_min[index_theta, index_phi] = x_min
            data_y_shear_min[index_theta, index_phi] = y_min
            data_z_shear_min[index_theta, index_phi] = z_min

            shear_max = shear[1]
            z_max = shear_max * z
            x_max = shear_max * x
            y_max = shear_max * y
            data_x_shear_max[index_theta, index_phi] = x_max
            data_y_shear_max[index_theta, index_phi] = y_max
            data_z_shear_max[index_theta, index_phi] = z_max

    shear_min_average = np.average(data_shear_min)
    shear_min_min = np.min(data_shear_min)
    shear_min_max = np.max(data_shear_min)

    shear_max_average = np.average(data_shear_max)
    shear_max_min = np.min(data_shear_max)
    shear_max_max = np.max(data_shear_max)

    plt.figure()
    axes = plt.axes(projection='3d')

    norm_min = colors.Normalize(vmin=shear_min_min, vmax=shear_min_max, clip=False)
    norm_max = colors.Normalize(vmin=shear_max_min, vmax=shear_max_max, clip=False)

    axes.plot_surface(data_x_shear_min, data_y_shear_min, data_z_shear_min, norm=norm_min, cmap=_symmetrical_greens)
    axes.plot_surface(data_x_shear_max, data_y_shear_max, data_z_shear_max, norm=norm_max, cmap=_symmetrical_blues,
                      alpha=0.5)

    scalarmap_shear_min = cm.ScalarMappable(cmap=_symmetrical_greens, norm=norm_min)
    scalarmap_shear_min.set_clim(shear_min_min, shear_min_max)

    scalarmap_shear_max = cm.ScalarMappable(cmap=_symmetrical_blues, norm=norm_max)
    scalarmap_shear_max.set_clim(shear_max_min, shear_max_max)

    cbar_min = plt.colorbar(scalarmap_shear_min, location="bottom", orientation="horizontal", fraction=0.06, pad=-0.1,
                            ticks=[shear_min_min, shear_min_average, shear_min_max])
    cbar_min.ax.tick_params(labelsize='large')
    cbar_min.set_label(r'directional shear modulus $G_{min}$ (MPa)', size=15, labelpad=20)

    cbar_max = plt.colorbar(scalarmap_shear_max, location="top", orientation="horizontal", fraction=0.06, pad=-0.1,
                            ticks=[shear_max_min, shear_max_average, shear_max_max])
    cbar_max.ax.tick_params(labelsize='large')
    cbar_max.set_label(r'directional shear modulus $G_{max}$ (MPa)', size=15, labelpad=20)

    axes.figure.axes[1].tick_params(axis="x", labelsize=20)
    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_shear_modulus.png", transparent=True)
    plt.show()


def plot_poisson_2d(stiffness_matrix: StiffnessTensor) -> None:
    """2D plotter for Poisson coefficient"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    poisson_xy = list(map(lambda x: stiffness_matrix.poisson_2d((np.pi/2.0, x)), theta_array))
    poisson_xz = list(map(lambda x: stiffness_matrix.poisson_2d((x, 0.0)), theta_array))
    poisson_yz = list(map(lambda x: stiffness_matrix.poisson_2d((x, np.pi/2.0)), theta_array))

    data_x_xy_1 = np.array([poisson[0] * np.cos(theta) for poisson, theta in zip(poisson_xy, theta_array)])
    data_y_xy_1 = np.array([poisson[0] * np.sin(theta) for poisson, theta in zip(poisson_xy, theta_array)])
    data_x_xy_2 = np.array([poisson[1] * np.cos(theta) for poisson, theta in zip(poisson_xy, theta_array)])
    data_y_xy_2 = np.array([poisson[1] * np.sin(theta) for poisson, theta in zip(poisson_xy, theta_array)])
    data_x_xy_3 = np.array([poisson[2] * np.cos(theta) for poisson, theta in zip(poisson_xy, theta_array)])
    data_y_xy_3 = np.array([poisson[2] * np.sin(theta) for poisson, theta in zip(poisson_xy, theta_array)])

    data_x_xz_1 = np.array([poisson[0] * np.sin(theta) for poisson, theta in zip(poisson_xz, theta_array)])
    data_y_xz_1 = np.array([poisson[0] * np.cos(theta) for poisson, theta in zip(poisson_xz, theta_array)])
    data_x_xz_2 = np.array([poisson[1] * np.sin(theta) for poisson, theta in zip(poisson_xz, theta_array)])
    data_y_xz_2 = np.array([poisson[1] * np.cos(theta) for poisson, theta in zip(poisson_xz, theta_array)])
    data_x_xz_3 = np.array([poisson[2] * np.sin(theta) for poisson, theta in zip(poisson_xz, theta_array)])
    data_y_xz_3 = np.array([poisson[2] * np.cos(theta) for poisson, theta in zip(poisson_xz, theta_array)])

    data_x_yz_1 = np.array([poisson[0] * np.sin(theta) for poisson, theta in zip(poisson_yz, theta_array)])
    data_y_yz_1 = np.array([poisson[0] * np.cos(theta) for poisson, theta in zip(poisson_yz, theta_array)])
    data_x_yz_2 = np.array([poisson[1] * np.sin(theta) for poisson, theta in zip(poisson_yz, theta_array)])
    data_y_yz_2 = np.array([poisson[1] * np.cos(theta) for poisson, theta in zip(poisson_yz, theta_array)])
    data_x_yz_3 = np.array([poisson[2] * np.sin(theta) for poisson, theta in zip(poisson_yz, theta_array)])
    data_y_yz_3 = np.array([poisson[2] * np.cos(theta) for poisson, theta in zip(poisson_yz, theta_array)])

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3)
    ax_xy.plot(data_x_xy_1, data_y_xy_1, 'b-')
    ax_xy.plot(data_x_xy_2, data_y_xy_2, 'g-')
    ax_xy.plot(data_x_xy_3, data_y_xy_3, 'r-')
    ax_xy.set_title("Poisson coefficient on (xy) plane")
    ax_xz.plot(data_x_xz_1, data_y_xz_1, 'b-')
    ax_xz.plot(data_x_xz_2, data_y_xz_2, 'g-')
    ax_xz.plot(data_x_xz_3, data_y_xz_3, 'r-')
    ax_xz.set_title("Poisson coefficient on (xz) plane")
    ax_yz.plot(data_x_yz_1, data_y_yz_1, 'b-')
    ax_yz.plot(data_x_yz_2, data_y_yz_2, 'g-')
    ax_yz.plot(data_x_yz_3, data_y_yz_3, 'g-')
    ax_yz.set_title("Poisson coefficient on (yz) plane")

    plt.savefig("planar_poisson_coefficient.png", transparent=True)
    plt.show()



def plot_poisson_3d(stiffness_matrix: StiffnessTensor) -> None:
    """3D plotter for Poisson coefficient"""

    n_points = 200

    theta_array = np.linspace(0.0, np.pi, n_points)
    phi_array = np.linspace(0.0, np.pi, n_points)
    phi_plus_pi_array = [phi_array[i] + np.pi for i in range(1, len(phi_array))]
    phi_array = np.append(phi_array, phi_plus_pi_array)

    data_x_poisson_1 = np.zeros((len(theta_array), len(phi_array)))
    data_y_poisson_1 = np.zeros((len(theta_array), len(phi_array)))
    data_z_poisson_1 = np.zeros((len(theta_array), len(phi_array)))
    data_x_poisson_2 = np.zeros((len(theta_array), len(phi_array)))
    data_y_poisson_2 = np.zeros((len(theta_array), len(phi_array)))
    data_z_poisson_2 = np.zeros((len(theta_array), len(phi_array)))
    data_x_poisson_3 = np.zeros((len(theta_array), len(phi_array)))
    data_y_poisson_3 = np.zeros((len(theta_array), len(phi_array)))
    data_z_poisson_3 = np.zeros((len(theta_array), len(phi_array)))

    data_poisson_1 = np.zeros((n_points, 2 * n_points))
    data_poisson_2 = np.zeros((n_points, 2 * n_points))
    data_poisson_3 = np.zeros((n_points, 2 * n_points))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):
            poisson = stiffness_matrix.poisson_3d((theta_array[index_theta], phi_array[index_phi]))
            z = np.cos(theta_array[index_theta])
            x = np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y = np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])

            poisson_1 = poisson[0]
            data_poisson_1[(theta_array[index_theta], phi_array[index_phi])] = poisson_1
            data_x_poisson_1[(theta_array[index_theta], phi_array[index_phi])] = poisson_1 * x
            data_y_poisson_1[(theta_array[index_theta], phi_array[index_phi])] = poisson_1 * y
            data_z_poisson_1[(theta_array[index_theta], phi_array[index_phi])] = poisson_1 * z

            poisson_2 = poisson[1]
            data_poisson_2[(theta_array[index_theta], phi_array[index_phi])] = poisson_2
            data_x_poisson_2[(theta_array[index_theta], phi_array[index_phi])] = poisson_2 * x
            data_y_poisson_2[(theta_array[index_theta], phi_array[index_phi])] = poisson_2 * y
            data_z_poisson_2[(theta_array[index_theta], phi_array[index_phi])] = poisson_2 * z

            poisson_3 = poisson[2]
            data_poisson_3[(theta_array[index_theta], phi_array[index_phi])] = poisson_3
            data_x_poisson_3[(theta_array[index_theta], phi_array[index_phi])] = poisson_3 * x
            data_y_poisson_3[(theta_array[index_theta], phi_array[index_phi])] = poisson_3 * y
            data_z_poisson_3[(theta_array[index_theta], phi_array[index_phi])] = poisson_3 * z

    poisson_1_average = np.average(data_poisson_1)
    poisson_1_min = np.min(data_poisson_1)
    poisson_1_max = np.max(data_poisson_1)

    poisson_2_average = np.average(data_poisson_2)
    poisson_2_min = np.min(data_poisson_2)
    poisson_2_max = np.max(data_poisson_2)

    poisson_3_average = np.average(data_poisson_3)
    poisson_3_min = np.min(data_poisson_3)
    poisson_3_max = np.max(data_poisson_3)

    plt.figure()
    axes = plt.axes(projection='3d')

    norm_poisson_1 = colors.Normalize(vmin=poisson_1_min, vmax=poisson_1_max, clip=False)
    norm_poisson_2 = colors.Normalize(vmin=poisson_2_min, vmax=poisson_2_max, clip=False)
    norm_poisson_3 = colors.Normalize(vmin=poisson_3_min, vmax=poisson_3_max, clip=False)

    axes.plot_surface(data_x_poisson_1, data_y_poisson_1, data_z_poisson_1, norm=norm_poisson_1,
                      cmap=_symmetrical_greens)
    axes.plot_surface(data_x_poisson_2, data_y_poisson_2, data_z_poisson_2, norm=norm_poisson_2,
                      cmap=_symmetrical_blues,
                      alpha=0.5)
    axes.plot_surface(data_x_poisson_3, data_y_poisson_3, data_z_poisson_3, norm=norm_poisson_3, cmap=_symmetrical_reds,
                      alpha=0.5)

    scalarmap_poisson_1 = cm.ScalarMappable(cmap=_symmetrical_greens, norm=norm_poisson_1)
    scalarmap_poisson_1.set_clim(poisson_1_min, poisson_1_max)

    scalarmap_poisson_2 = cm.ScalarMappable(cmap=_symmetrical_blues, norm=norm_poisson_2)
    scalarmap_poisson_2.set_clim(poisson_2_min, poisson_2_max)

    scalarmap_poisson_3 = cm.ScalarMappable(cmap=_symmetrical_reds, norm=norm_poisson_3)
    scalarmap_poisson_3.set_clim(poisson_3_min, poisson_3_max)

    cbar_poisson_1 = plt.colorbar(scalarmap_poisson_1, location="bottom", orientation="horizontal", fraction=0.06,
                                  pad=-0.1,
                                  ticks=[poisson_1_min, poisson_1_average, poisson_1_max])
    cbar_poisson_1.ax.tick_params(labelsize='large')
    cbar_poisson_1.set_label(r'Poisson coefficient ' + "\u03BD", size=15, labelpad=20)

    cbar_poisson_2 = plt.colorbar(scalarmap_poisson_2, location="bottom", orientation="horizontal", fraction=0.06,
                                  pad=-0.1,
                                  ticks=[poisson_2_min, poisson_2_average, poisson_2_max])
    cbar_poisson_2.ax.tick_params(labelsize='large')
    cbar_poisson_2.set_label(r'Poisson coefficient ' + "\u03BD", size=15, labelpad=20)

    cbar_poisson_3 = plt.colorbar(scalarmap_poisson_3, location="bottom", orientation="horizontal", fraction=0.06,
                                  pad=-0.1,
                                  ticks=[poisson_3_min, poisson_3_average, poisson_3_max])
    cbar_poisson_3.ax.tick_params(labelsize='large')
    cbar_poisson_3.set_label(r'Poisson coefficient ' + "\u03BD", size=15, labelpad=20)

    axes.figure.axes[1].tick_params(axis="x", labelsize=20)
    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_poisson_coefficient.png", transparent=True)
    plt.show()
