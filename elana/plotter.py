import numpy as np
from elana.abstract_stiffness_tensor import AbstractStiffnessTensor
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib import cm, colors

def plot_young_2d(stiffness_matrix: AbstractStiffnessTensor) -> None:
    """2D plotter for Young modulus"""

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(55, 15))
    ax_xy.plot(stiffness_matrix.data_young_x_xy, stiffness_matrix.data_young_y_xy, 'g-')
    ax_xy.grid()
    ax_xy.set_title("Young modulus on (xy) plane")
    ax_xz.plot(stiffness_matrix.data_young_x_xz, stiffness_matrix.data_young_y_xz, 'g-')
    ax_xz.grid()
    ax_xz.set_title("Young modulus on (xz) plane")
    ax_yz.plot(stiffness_matrix.data_young_x_yz, stiffness_matrix.data_young_y_yz, 'g-')
    ax_yz.set_title("Young modulus on (yz) plane")

    plt.savefig("planar_young.png")
    plt.show()


def plot_young_3d(stiffness_matrix: AbstractStiffnessTensor) -> None:
    """3D plotter for Young modulus"""

    plt.figure()
    axes = plt.axes(projection='3d')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    axes.set_title(r'Directional stiffness $E$ (MPa)')

    norm = colors.Normalize(vmin=stiffness_matrix.young_3d_min, vmax=stiffness_matrix.young_3d_max, clip=False)

    scalarmap = cm.ScalarMappable(cmap='summer', norm=norm)
    scalarmap.set_clim(stiffness_matrix.young_3d_min, stiffness_matrix.young_3d_max)
    scalarmap.set_array([])
    fcolors = scalarmap.to_rgba(stiffness_matrix.data_young_3d)


    cbar = plt.colorbar(scalarmap, orientation="horizontal", fraction=0.05, pad=0.1,
                        ticks=[stiffness_matrix.young_3d_min, stiffness_matrix.young_3d_average, stiffness_matrix.young_3d_max])
    cbar.ax.tick_params(labelsize='large')

    axes.plot_surface(stiffness_matrix.data_young_3d_x, stiffness_matrix.data_young_3d_y, stiffness_matrix.data_young_3d_z, facecolors=fcolors, norm=norm, cmap='summer', linewidth=0.1, edgecolor = 'k', alpha=0.8)
    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_young.png")
    plt.show()


def plot_linear_compressibility_2d(stiffness_matrix: AnisotropicStiffnessTensor) -> None:
    """2D plotter for linear compressibility modulus"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    linear_compressibility_pos_xy = list(
        map(lambda x: max(0.0, stiffness_matrix.linear_compressibility((np.pi / 2.0, x))), theta_array))
    linear_compressibility_pos_xz = list(
        map(lambda x: max(0.0, stiffness_matrix.linear_compressibility((x, 0.0))), theta_array))
    linear_compressibility_pos_yz = list(
        map(lambda x: max(0.0, stiffness_matrix.linear_compressibility((x, np.pi / 2.0))), theta_array))

    data_x_xy_pos, data_y_xy_pos = make_planar_plot_data(linear_compressibility_pos_xy * np.cos(theta_array),
                                                         linear_compressibility_pos_xy * np.sin(theta_array))
    data_x_xz_pos, data_y_xz_pos = make_planar_plot_data(linear_compressibility_pos_xz * np.sin(theta_array),
                                                         linear_compressibility_pos_xz * np.cos(theta_array))
    data_x_yz_pos, data_y_yz_pos = make_planar_plot_data(linear_compressibility_pos_yz * np.sin(theta_array),
                                                         linear_compressibility_pos_yz * np.cos(theta_array))

    linear_compressibility_neg_xy = list(
        map(lambda x: max(0.0, -stiffness_matrix.linear_compressibility((np.pi / 2.0, x))), theta_array))
    linear_compressibility_neg_xz = list(
        map(lambda x: max(0.0, -stiffness_matrix.linear_compressibility((x, 0.0))), theta_array))
    linear_compressibility_neg_yz = list(
        map(lambda x: max(0.0, -stiffness_matrix.linear_compressibility((x, np.pi / 2.0))), theta_array))

    data_x_xy_neg, data_y_xy_neg = make_planar_plot_data(linear_compressibility_neg_xy * np.cos(theta_array),
                                                         linear_compressibility_neg_xy * np.sin(theta_array))
    data_x_xz_neg, data_y_xz_neg = make_planar_plot_data(linear_compressibility_neg_xz * np.sin(theta_array),
                                                         linear_compressibility_neg_xz * np.cos(theta_array))
    data_x_yz_neg, data_y_yz_neg = make_planar_plot_data(linear_compressibility_neg_yz * np.sin(theta_array),
                                                         linear_compressibility_neg_yz * np.cos(theta_array))

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(55, 15))
    ax_xy.plot(data_x_xy_pos, data_y_xy_pos, 'g-')
    ax_xy.plot(data_x_xy_neg, data_y_xy_neg, 'r-')
    ax_xy.grid()
    ax_xy.set_title("Linear compressibility on (xy) plane")
    ax_xz.plot(data_x_xz_pos, data_y_xz_pos, 'g-')
    ax_xz.plot(data_x_xz_neg, data_y_xz_neg, 'r-')
    ax_xz.grid()
    ax_xz.set_title("Linear compressibility on (xz) plane")
    ax_yz.plot(data_x_yz_pos, data_y_yz_pos, 'g-')
    ax_yz.plot(data_x_yz_neg, data_y_yz_neg, 'r-')
    ax_yz.grid()
    ax_yz.set_title("Linear compressibility on (yz) plane")

    plt.savefig("planar_linear_compressibility.png")
    plt.show()


def plot_linear_compressibility_3d(stiffness_matrix: AnisotropicStiffnessTensor) -> None:
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

    linear_compressibility_pos_max = np.max(data_linear_compressibility_pos)
    linear_compressibility_pos_min = np.min(data_linear_compressibility_pos)
    linear_compressibility_neg_max = np.max(data_linear_compressibility_neg)
    linear_compressibility_neg_min = np.min(data_linear_compressibility_neg)

    plt.figure(figsize=(10,10))
    axes = plt.axes(projection='3d')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    axes.set_title(r'Directional linear compressibility $LC$ (TPa $^{-1}$)')

    norm_pos = colors.Normalize(vmin=linear_compressibility_pos_min, vmax=linear_compressibility_pos_max, clip=False)
    norm_neg = colors.Normalize(vmin=linear_compressibility_neg_min, vmax=linear_compressibility_neg_max, clip=False)

    scalarmap_pos = cm.ScalarMappable(cmap='Greens', norm=norm_pos)
    scalarmap_pos.set_clim(linear_compressibility_pos_min, linear_compressibility_pos_max)
    scalarmap_pos.set_array([])
    fcolors_pos = scalarmap_pos.to_rgba(data_linear_compressibility_pos)

    cbar_pos = plt.colorbar(scalarmap_pos, orientation="horizontal", pad=0.05, shrink=0.6,
                        ticks=[linear_compressibility_pos_min, linear_compressibility_pos_max])
    cbar_pos.ax.tick_params(labelsize='large')

    scalarmap_neg = cm.ScalarMappable(cmap='Reds', norm=norm_neg)
    scalarmap_neg.set_clim(linear_compressibility_neg_min, linear_compressibility_neg_max)
    scalarmap_neg.set_array([])
    fcolors_neg = scalarmap_neg.to_rgba(data_linear_compressibility_neg)

    cbar_neg = plt.colorbar(scalarmap_neg, orientation="horizontal", pad=0.08, shrink=0.6,
                        ticks=[linear_compressibility_neg_min, linear_compressibility_neg_max])
    cbar_neg.ax.tick_params(labelsize='large')

    axes.plot_surface(data_x_pos, data_y_pos, data_z_pos, facecolors=fcolors_pos, norm=norm_pos, cmap='Greens', linewidth=0.1, edgecolor = 'k', alpha=0.8)
    axes.plot_surface(data_x_neg, data_y_neg, data_z_neg, facecolors=fcolors_neg, norm=norm_neg, cmap='Reds', linewidth=0.1, edgecolor = 'k', alpha=0.8)

    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_linear_compressibility.png")
    plt.show()


def plot_shear_modulus_2d(stiffness_matrix: AnisotropicStiffnessTensor) -> None:
    """2D plotter for shear modulus"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    shear_xy = list(map(lambda x: stiffness_matrix.shear_2d((np.pi / 2.0, x)), theta_array))
    shear_xz = list(map(lambda x: stiffness_matrix.shear_2d((x, 0.0)), theta_array))
    shear_yz = list(map(lambda x: stiffness_matrix.shear_2d((x, np.pi / 2.0)), theta_array))

    data_x_xy_min, data_y_xy_min = make_planar_plot_data(
        np.array([shear[0] * np.cos(theta) for shear, theta in zip(shear_xy, theta_array)]),
        np.array([shear[0] * np.sin(theta) for shear, theta in zip(shear_xy, theta_array)]))
    data_x_xy_max, data_y_xy_max = make_planar_plot_data(
        np.array([shear[1] * np.cos(theta) for shear, theta in zip(shear_xy, theta_array)]),
        np.array([shear[1] * np.sin(theta) for shear, theta in zip(shear_xy, theta_array)]))

    data_x_xz_min, data_y_xz_min = make_planar_plot_data(
        np.array([shear[0] * np.sin(theta) for shear, theta in zip(shear_xz, theta_array)]),
        np.array([shear[0] * np.cos(theta) for shear, theta in zip(shear_xz, theta_array)]))
    data_x_xz_max, data_y_xz_max = make_planar_plot_data(
        np.array([shear[1] * np.sin(theta) for shear, theta in zip(shear_xz, theta_array)]),
        np.array([shear[1] * np.cos(theta) for shear, theta in zip(shear_xz, theta_array)]))

    data_x_yz_min, data_y_yz_min = make_planar_plot_data(
        np.array([shear[0] * np.sin(theta) for shear, theta in zip(shear_yz, theta_array)]),
        np.array([shear[0] * np.cos(theta) for shear, theta in zip(shear_yz, theta_array)]))
    data_x_yz_max, data_y_yz_max = make_planar_plot_data(
        np.array([shear[1] * np.sin(theta) for shear, theta in zip(shear_yz, theta_array)]),
        np.array([shear[1] * np.cos(theta) for shear, theta in zip(shear_yz, theta_array)]))

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(55, 15))
    ax_xy.plot(data_x_xy_max, data_y_xy_max, 'b-')
    ax_xy.plot(data_x_xy_min, data_y_xy_min, 'g-')
    ax_xy.grid()
    ax_xy.set_title("Shear modulus on (xy) plane")
    ax_xz.plot(data_x_xz_max, data_y_xz_max, 'b-')
    ax_xz.plot(data_x_xz_min, data_y_xz_min, 'g-')
    ax_xz.grid()
    ax_xz.set_title("Shear modulus on (xz) plane")
    ax_yz.plot(data_x_yz_max, data_y_yz_max, 'b-')
    ax_yz.plot(data_x_yz_min, data_y_yz_min, 'g-')
    ax_yz.grid()
    ax_yz.set_title("Shear modulus on (yz) plane")

    plt.savefig("planar_shear_modulus.png")
    plt.show()


def plot_shear_modulus_3d(stiffness_matrix: AnisotropicStiffnessTensor) -> None:
    """3D plotter for shear modulus"""

    n_points = 100

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

    plt.figure(figsize=(10,10))
    axes = plt.axes(projection='3d')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    axes.set_title(r'Directional shear modulus $G_{max}$ (MPa)')

    norm_min = colors.Normalize(vmin=shear_min_min, vmax=shear_min_max, clip=False)
    norm_max = colors.Normalize(vmin=shear_max_min, vmax=shear_max_max, clip=False)

    scalarmap_shear_min = cm.ScalarMappable(cmap="Greens", norm=norm_min)
    scalarmap_shear_min.set_clim(shear_min_min, shear_min_max)
    scalarmap_shear_min.set_array([])
    fcolors_shear_min = scalarmap_shear_min.to_rgba(data_shear_min)

    scalarmap_shear_max = cm.ScalarMappable(cmap="Blues", norm=norm_max)
    scalarmap_shear_max.set_clim(shear_max_min, shear_max_max)
    scalarmap_shear_max.set_array([])
    fcolors_shear_max = scalarmap_shear_max.to_rgba(data_shear_max)

    axes.plot_surface(data_x_shear_min, data_y_shear_min, data_z_shear_min, facecolors=fcolors_shear_min, norm=norm_min, cmap="Greens", linewidth=0.05, edgecolor = 'k', alpha=0.8)
    axes.plot_surface(data_x_shear_max, data_y_shear_max, data_z_shear_max, facecolors=fcolors_shear_max, norm=norm_max, cmap="Blues", linewidth=0.05, edgecolor = 'k',
                      alpha=0.5)

    cbar_min = plt.colorbar(scalarmap_shear_min, orientation="horizontal", pad=0.05, shrink=0.6,
                            ticks=[shear_min_min, shear_min_average, shear_min_max])
    cbar_min.ax.tick_params(labelsize='large')

    cbar_max = plt.colorbar(scalarmap_shear_max, orientation="horizontal", pad=0.08, shrink=0.6,
                            ticks=[shear_max_min, shear_max_average, shear_max_max])
    cbar_max.ax.tick_params(labelsize='large')

    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_shear_modulus.png")
    plt.show()


def plot_poisson_2d(stiffness_matrix: AnisotropicStiffnessTensor) -> None:
    """2D plotter for Poisson coefficient"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    poisson_xy = list(map(lambda x: stiffness_matrix.poisson_2d((np.pi / 2.0, x)), theta_array))
    poisson_xz = list(map(lambda x: stiffness_matrix.poisson_2d((x, 0.0)), theta_array))
    poisson_yz = list(map(lambda x: stiffness_matrix.poisson_2d((x, np.pi / 2.0)), theta_array))

    data_x_xy_1, data_y_xy_1 = make_planar_plot_data(np.array([poisson[0] * np.cos(theta) for poisson, theta in zip(poisson_xy, theta_array)]), np.array([poisson[0] * np.sin(theta) for poisson, theta in zip(poisson_xy, theta_array)]))
    data_x_xy_2, data_y_xy_2 = make_planar_plot_data(np.array([poisson[1] * np.cos(theta) for poisson, theta in zip(poisson_xy, theta_array)]), np.array([poisson[1] * np.sin(theta) for poisson, theta in zip(poisson_xy, theta_array)]))
    data_x_xy_3, data_y_xy_3 = make_planar_plot_data(np.array([poisson[2] * np.cos(theta) for poisson, theta in zip(poisson_xy, theta_array)]), np.array([poisson[2] * np.sin(theta) for poisson, theta in zip(poisson_xy, theta_array)]))

    data_x_xz_1, data_y_xz_1 = make_planar_plot_data(np.array([poisson[0] * np.sin(theta) for poisson, theta in zip(poisson_xz, theta_array)]), np.array([poisson[0] * np.cos(theta) for poisson, theta in zip(poisson_xz, theta_array)]))
    data_x_xz_2, data_y_xz_2 = make_planar_plot_data(np.array([poisson[1] * np.sin(theta) for poisson, theta in zip(poisson_xz, theta_array)]), np.array([poisson[1] * np.cos(theta) for poisson, theta in zip(poisson_xz, theta_array)]))
    data_x_xz_3, data_y_xz_3 = make_planar_plot_data(np.array([poisson[2] * np.sin(theta) for poisson, theta in zip(poisson_xz, theta_array)]), np.array([poisson[2] * np.cos(theta) for poisson, theta in zip(poisson_xz, theta_array)]))

    data_x_yz_1, data_y_yz_1 = make_planar_plot_data(np.array([poisson[0] * np.sin(theta) for poisson, theta in zip(poisson_yz, theta_array)]), np.array([poisson[0] * np.cos(theta) for poisson, theta in zip(poisson_yz, theta_array)]))
    data_x_yz_2, data_y_yz_2 = make_planar_plot_data(np.array([poisson[1] * np.sin(theta) for poisson, theta in zip(poisson_yz, theta_array)]), np.array([poisson[1] * np.cos(theta) for poisson, theta in zip(poisson_yz, theta_array)]))
    data_x_yz_3, data_y_yz_3 = make_planar_plot_data(np.array([poisson[2] * np.sin(theta) for poisson, theta in zip(poisson_yz, theta_array)]), np.array([poisson[2] * np.cos(theta) for poisson, theta in zip(poisson_yz, theta_array)]))

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(55, 15))
    ax_xy.plot(data_x_xy_1, data_y_xy_1, 'r-')
    ax_xy.plot(data_x_xy_2, data_y_xy_2, 'g-')
    ax_xy.plot(data_x_xy_3, data_y_xy_3, 'b-')
    ax_xy.grid()
    ax_xy.set_title("Poisson coefficient on (xy) plane")
    ax_xz.plot(data_x_xz_1, data_y_xz_1, 'r-')
    ax_xz.plot(data_x_xz_2, data_y_xz_2, 'g-')
    ax_xz.plot(data_x_xz_3, data_y_xz_3, 'b-')
    ax_xz.grid()
    ax_xz.set_title("Poisson coefficient on (xz) plane")
    ax_yz.plot(data_x_yz_1, data_y_yz_1, 'r-')
    ax_yz.plot(data_x_yz_2, data_y_yz_2, 'g-')
    ax_yz.plot(data_x_yz_3, data_y_yz_3, 'b-')
    ax_yz.plot()
    ax_yz.set_title("Poisson coefficient on (yz) plane")
    plt.grid()

    plt.savefig("planar_poisson_coefficient.png")
    plt.show()


def plot_poisson_3d(stiffness_matrix: AnisotropicStiffnessTensor) -> None:
    """3D plotter for Poisson coefficient"""

    n_points = 50

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
            data_poisson_1[index_theta, index_phi] = poisson_1
            data_x_poisson_1[index_theta, index_phi] = poisson_1 * x
            data_y_poisson_1[index_theta, index_phi] = poisson_1 * y
            data_z_poisson_1[index_theta, index_phi] = poisson_1 * z

            poisson_2 = poisson[1]
            data_poisson_2[index_theta, index_phi] = poisson_2
            data_x_poisson_2[index_theta, index_phi] = poisson_2 * x
            data_y_poisson_2[index_theta, index_phi] = poisson_2 * y
            data_z_poisson_2[index_theta, index_phi] = poisson_2 * z

            poisson_3 = poisson[2]
            data_poisson_3[index_theta, index_phi] = poisson_3
            data_x_poisson_3[index_theta, index_phi] = poisson_3 * x
            data_y_poisson_3[index_theta, index_phi] = poisson_3 * y
            data_z_poisson_3[index_theta, index_phi] = poisson_3 * z

    poisson_1_average = np.average(data_poisson_1)
    poisson_1_min = np.min(data_poisson_1)
    poisson_1_max = np.max(data_poisson_1)

    poisson_2_average = np.average(data_poisson_2)
    poisson_2_min = np.min(data_poisson_2)
    poisson_2_max = np.max(data_poisson_2)

    poisson_3_average = np.average(data_poisson_3)
    poisson_3_min = np.min(data_poisson_3)
    poisson_3_max = np.max(data_poisson_3)

    plt.figure(figsize=(12,12))
    axes = plt.axes(projection='3d')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    axes.set_title(r'Directional Poisson coefficient ' + "\u03BD")

    norm_poisson_1 = colors.Normalize(vmin=poisson_1_min, vmax=poisson_1_max, clip=False)
    norm_poisson_2 = colors.Normalize(vmin=poisson_2_min, vmax=poisson_2_max, clip=False)
    norm_poisson_3 = colors.Normalize(vmin=poisson_3_min, vmax=poisson_3_max, clip=False)

    scalarmap_poisson_1 = cm.ScalarMappable(cmap="Reds", norm=norm_poisson_1)
    scalarmap_poisson_1.set_clim(poisson_1_min, poisson_1_max)
    scalarmap_poisson_1.set_array([])
    fcolors_poisson_1 = scalarmap_poisson_1.to_rgba(data_poisson_1)

    scalarmap_poisson_2 = cm.ScalarMappable(cmap="Greens", norm=norm_poisson_2)
    scalarmap_poisson_2.set_clim(poisson_2_min, poisson_2_max)
    scalarmap_poisson_2.set_array([])
    fcolors_poisson_2 = scalarmap_poisson_2.to_rgba(data_poisson_2)

    scalarmap_poisson_3 = cm.ScalarMappable(cmap="Blues", norm=norm_poisson_3)
    scalarmap_poisson_3.set_clim(poisson_3_min, poisson_3_max)
    scalarmap_poisson_3.set_array([])
    fcolors_poisson_3 = scalarmap_poisson_3.to_rgba(data_poisson_3)

    axes.plot_surface(data_x_poisson_1, data_y_poisson_1, data_z_poisson_1, facecolors=fcolors_poisson_1, norm=norm_poisson_1,
                      cmap="Reds", linewidth=0.1, edgecolor = 'k')
    axes.plot_surface(data_x_poisson_2, data_y_poisson_2, data_z_poisson_2, facecolors=fcolors_poisson_2, norm=norm_poisson_2,
                      cmap="Greens",
                      alpha=0.5, linewidth=0.1, edgecolor = 'k')
    axes.plot_surface(data_x_poisson_3, data_y_poisson_3, data_z_poisson_3, facecolors=fcolors_poisson_3, norm=norm_poisson_3, cmap="Blues",
                      alpha=0.5, linewidth=0.1, edgecolor = 'k')

    cbar_poisson_1 = plt.colorbar(scalarmap_poisson_1, pad=0.06, orientation="horizontal", shrink=0.6,
                                  ticks=[poisson_1_min, poisson_1_average, poisson_1_max])
    cbar_poisson_1.ax.tick_params(labelsize='large')

    cbar_poisson_2 = plt.colorbar(scalarmap_poisson_2, pad=0.07, orientation="horizontal", shrink=0.6,
                                  ticks=[poisson_2_min, poisson_2_average, poisson_2_max])
    cbar_poisson_2.ax.tick_params(labelsize='large')

    cbar_poisson_3 = plt.colorbar(scalarmap_poisson_3, pad=0.075, orientation="horizontal", shrink=0.6,
                                  ticks=[poisson_3_min, poisson_3_average, poisson_3_max])
    cbar_poisson_3.ax.tick_params(labelsize='large')

    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_poisson_coefficient.png")
    plt.show()
