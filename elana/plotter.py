import numpy as np
from elana.abstract_stiffness_tensor import AbstractStiffnessTensor
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from elana.operations import make_planar_plot_data

def plot_young_2d(stiffness_matrix: AbstractStiffnessTensor) -> None:
    """2D plotter for Young modulus"""
    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    data_young_x_xy, data_young_y_xy = make_planar_plot_data(stiffness_matrix.data_young_2d["xy"] * np.cos(theta_array),
                                                             stiffness_matrix.data_young_2d["xy"] * np.sin(theta_array))
    data_young_x_xz, data_young_y_xz = make_planar_plot_data(stiffness_matrix.data_young_2d["xz"] * np.sin(theta_array),
                                                             stiffness_matrix.data_young_2d["xz"] * np.cos(theta_array))
    data_young_x_yz, data_young_y_yz = make_planar_plot_data(stiffness_matrix.data_young_2d["yz"] * np.sin(theta_array),
                                                             stiffness_matrix.data_young_2d["yz"] * np.cos(theta_array))

    fig, (ax_xy, ax_xz, ax_yz) = plt.subplots(1, 3, figsize=(55, 15))

    ax_xy.plot(data_young_x_xy, data_young_y_xy, 'g-')
    ax_xy.grid()
    ax_xy.set_title("Young modulus on (xy) plane")

    ax_xz.plot(data_young_x_xz, data_young_y_xz, 'g-')
    ax_xz.grid()
    ax_xz.set_title("Young modulus on (xz) plane")

    ax_yz.plot(data_young_x_yz, data_young_y_yz, 'g-')
    ax_yz.grid()
    ax_yz.set_title("Young modulus on (yz) plane")

    plt.savefig("planar_young.png")
    plt.show()


def plot_young_3d(stiffness_matrix: AbstractStiffnessTensor) -> None:
    """3D plotter for Young modulus"""

    n_points = 200

    theta_array = np.linspace(0.0, np.pi, n_points)
    phi_array = np.linspace(0.0, 2 * np.pi, 2 * n_points)

    data_xyz = np.zeros((3,n_points, 2*n_points))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):
            z = stiffness_matrix.data_young_3d[index_theta, index_phi] * np.cos(theta_array[index_theta])
            x = stiffness_matrix.data_young_3d[index_theta, index_phi] * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y = stiffness_matrix.data_young_3d[index_theta, index_phi] * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            data_xyz[0,index_theta, index_phi] = x
            data_xyz[1,index_theta, index_phi] = y
            data_xyz[2,index_theta, index_phi] = z

    young_3d_min = np.min(stiffness_matrix.data_young_3d)
    young_3d_max = np.max(stiffness_matrix.data_young_3d)
    young_3d_average = np.mean(stiffness_matrix.data_young_3d)

    plt.figure()
    axes = plt.axes(projection='3d')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    axes.set_title(r'Directional stiffness $E$ (MPa)')

    norm = colors.Normalize(vmin=young_3d_min, vmax=young_3d_max, clip=False)

    scalarmap = cm.ScalarMappable(cmap='summer', norm=norm)
    scalarmap.set_clim(young_3d_min, young_3d_max)
    scalarmap.set_array([])
    fcolors = scalarmap.to_rgba(stiffness_matrix.data_young_3d)


    cbar = plt.colorbar(scalarmap, orientation="horizontal", fraction=0.05, pad=0.1,
                        ticks=[young_3d_min, young_3d_average, young_3d_max])
    cbar.ax.tick_params(labelsize='large')

    axes.plot_surface(data_xyz[0,:,:], data_xyz[1,:,:], data_xyz[2,:,:], facecolors=fcolors, norm=norm, cmap='summer', linewidth=0.1, edgecolor = 'k', alpha=0.8)
    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_young.png")
    plt.show()


def plot_linear_compressibility_2d(stiffness_matrix: AbstractStiffnessTensor) -> None:
    """2D plotter for linear compressibility modulus"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    data_x_xy_pos, data_y_xy_pos = make_planar_plot_data(stiffness_matrix.data_linear_compressibility_2d["pos"]["xy"] * np.cos(theta_array),
                                                         stiffness_matrix.data_linear_compressibility_2d["pos"]["xy"] * np.sin(theta_array))
    data_x_xz_pos, data_y_xz_pos = make_planar_plot_data(stiffness_matrix.data_linear_compressibility_2d["pos"]["xz"] * np.sin(theta_array),
                                                         stiffness_matrix.data_linear_compressibility_2d["pos"]["xz"] * np.cos(theta_array))
    data_x_yz_pos, data_y_yz_pos = make_planar_plot_data(stiffness_matrix.data_linear_compressibility_2d["pos"]["yz"] * np.sin(theta_array),
                                                         stiffness_matrix.data_linear_compressibility_2d["pos"]["yz"] * np.cos(theta_array))

    data_x_xy_neg, data_y_xy_neg = make_planar_plot_data(stiffness_matrix.data_linear_compressibility_2d["neg"]["xy"] * np.cos(theta_array),
                                                         stiffness_matrix.data_linear_compressibility_2d["neg"]["xy"] * np.sin(theta_array))
    data_x_xz_neg, data_y_xz_neg = make_planar_plot_data(stiffness_matrix.data_linear_compressibility_2d["neg"]["xz"] * np.sin(theta_array),
                                                         stiffness_matrix.data_linear_compressibility_2d["neg"]["xz"] * np.cos(theta_array))
    data_x_yz_neg, data_y_yz_neg = make_planar_plot_data(stiffness_matrix.data_linear_compressibility_2d["neg"]["yz"] * np.sin(theta_array),
                                                         stiffness_matrix.data_linear_compressibility_2d["neg"]["yz"] * np.cos(theta_array))

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


def plot_linear_compressibility_3d(stiffness_matrix: AbstractStiffnessTensor) -> None:
    """3D plotter for linear compressibility modulus"""

    n_points = 200

    theta_array = np.linspace(0.0, np.pi, n_points)
    phi_array = np.linspace(0.0, 2 * np.pi, 2 * n_points)

    data_xyz_pos = np.zeros((3, len(theta_array), len(phi_array)))
    data_xyz_neg = np.zeros((3, len(theta_array), len(phi_array)))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):

            x_pos = stiffness_matrix.data_linear_compressibility_3d["pos"][index_theta, index_phi] * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y_pos = stiffness_matrix.data_linear_compressibility_3d["pos"][index_theta, index_phi] * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            z_pos = stiffness_matrix.data_linear_compressibility_3d["pos"][index_theta, index_phi] * np.cos(theta_array[index_theta])

            x_neg = stiffness_matrix.data_linear_compressibility_3d["neg"][index_theta, index_phi] * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y_neg = stiffness_matrix.data_linear_compressibility_3d["neg"][index_theta, index_phi] * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            z_neg = stiffness_matrix.data_linear_compressibility_3d["neg"][index_theta, index_phi] * np.cos(theta_array[index_theta])

            data_xyz_pos[0, index_theta, index_phi] = x_pos
            data_xyz_pos[1, index_theta, index_phi] = y_pos
            data_xyz_pos[2, index_theta, index_phi] = z_pos

            data_xyz_neg[0, index_theta, index_phi] = x_neg
            data_xyz_neg[1, index_theta, index_phi] = y_neg
            data_xyz_neg[2, index_theta, index_phi] = z_neg

    linear_compressibility_pos_max = np.max(stiffness_matrix.data_linear_compressibility_3d["pos"])
    linear_compressibility_pos_min = np.min(stiffness_matrix.data_linear_compressibility_3d["pos"])
    linear_compressibility_neg_max = np.max(stiffness_matrix.data_linear_compressibility_3d["neg"])
    linear_compressibility_neg_min = np.min(stiffness_matrix.data_linear_compressibility_3d["neg"])

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
    fcolors_pos = scalarmap_pos.to_rgba(stiffness_matrix.data_linear_compressibility_3d["pos"])

    cbar_pos = plt.colorbar(scalarmap_pos, orientation="horizontal", pad=0.05, shrink=0.6,
                        ticks=[linear_compressibility_pos_min, linear_compressibility_pos_max])
    cbar_pos.ax.tick_params(labelsize='large')

    scalarmap_neg = cm.ScalarMappable(cmap='Reds', norm=norm_neg)
    scalarmap_neg.set_clim(linear_compressibility_neg_min, linear_compressibility_neg_max)
    scalarmap_neg.set_array([])
    fcolors_neg = scalarmap_neg.to_rgba(stiffness_matrix.data_linear_compressibility_3d["neg"])

    cbar_neg = plt.colorbar(scalarmap_neg, orientation="horizontal", pad=0.08, shrink=0.6,
                        ticks=[linear_compressibility_neg_min, linear_compressibility_neg_max])
    cbar_neg.ax.tick_params(labelsize='large')

    axes.plot_surface(data_xyz_pos[0,:,:], data_xyz_pos[1,:,:], data_xyz_pos[2,:,:], facecolors=fcolors_pos, norm=norm_pos, cmap='Greens', linewidth=0.1, edgecolor = 'k', alpha=0.8)
    axes.plot_surface(data_xyz_neg[0,:,:], data_xyz_neg[1,:,:], data_xyz_neg[2,:,:], facecolors=fcolors_neg, norm=norm_neg, cmap='Reds', linewidth=0.1, edgecolor = 'k', alpha=0.8)

    axes.azim = 30
    axes.elev = 30

    plt.savefig("directional_linear_compressibility.png")
    plt.show()


def plot_shear_modulus_2d(stiffness_matrix: AbstractStiffnessTensor) -> None:
    """2D plotter for shear modulus"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    data_x_xy_min, data_y_xy_min = make_planar_plot_data(
        np.array([shear[0] * np.cos(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["xy"], theta_array)]),
        np.array([shear[0] * np.sin(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["xy"], theta_array)]))
    data_x_xy_max, data_y_xy_max = make_planar_plot_data(
        np.array([shear[1] * np.cos(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["xy"], theta_array)]),
        np.array([shear[1] * np.sin(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["xy"], theta_array)]))

    data_x_xz_min, data_y_xz_min = make_planar_plot_data(
        np.array([shear[0] * np.sin(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["xz"], theta_array)]),
        np.array([shear[0] * np.cos(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["xz"], theta_array)]))
    data_x_xz_max, data_y_xz_max = make_planar_plot_data(
        np.array([shear[1] * np.sin(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["xz"], theta_array)]),
        np.array([shear[1] * np.cos(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["xz"], theta_array)]))

    data_x_yz_min, data_y_yz_min = make_planar_plot_data(
        np.array([shear[0] * np.sin(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["yz"], theta_array)]),
        np.array([shear[0] * np.cos(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["yz"], theta_array)]))
    data_x_yz_max, data_y_yz_max = make_planar_plot_data(
        np.array([shear[1] * np.sin(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["yz"], theta_array)]),
        np.array([shear[1] * np.cos(theta) for shear, theta in zip(stiffness_matrix.data_shear_2d["yz"], theta_array)]))

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


def plot_shear_modulus_3d(stiffness_matrix: AbstractStiffnessTensor) -> None:
    """3D plotter for shear modulus"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)
    phi_array = np.linspace(0.0, np.pi, n_points)
    phi_plus_pi_array = [phi_array[i] + np.pi for i in range(1, len(phi_array))]
    phi_array = np.append(phi_array, phi_plus_pi_array)

    data_xyz_shear_min = np.zeros((3, len(theta_array), len(phi_array)))
    data_xyz_shear_max = np.zeros((3, len(theta_array), len(phi_array)))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):

            shear_min = stiffness_matrix.data_shear_3d["min"][index_theta, index_phi]
            data_xyz_shear_min[0, index_theta, index_phi] = shear_min * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            data_xyz_shear_min[1, index_theta, index_phi] = shear_min * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            data_xyz_shear_min[2, index_theta, index_phi] = shear_min * np.cos(theta_array[index_theta])

            shear_max = stiffness_matrix.data_shear_3d["max"][index_theta, index_phi]
            data_xyz_shear_max[0, index_theta, index_phi] = shear_max * np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            data_xyz_shear_max[1, index_theta, index_phi] = shear_max * np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])
            data_xyz_shear_max[2, index_theta, index_phi] = shear_max * np.cos(theta_array[index_theta])

    shear_min_average = np.average(stiffness_matrix.data_shear_3d["min"])
    shear_min_min = np.min(stiffness_matrix.data_shear_3d["min"])
    shear_min_max = np.max(stiffness_matrix.data_shear_3d["min"])

    shear_max_average = np.average(stiffness_matrix.data_shear_3d["max"])
    shear_max_min = np.min(stiffness_matrix.data_shear_3d["max"])
    shear_max_max = np.max(stiffness_matrix.data_shear_3d["max"])

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
    fcolors_shear_min = scalarmap_shear_min.to_rgba(stiffness_matrix.data_shear_3d["min"])

    scalarmap_shear_max = cm.ScalarMappable(cmap="Blues", norm=norm_max)
    scalarmap_shear_max.set_clim(shear_max_min, shear_max_max)
    scalarmap_shear_max.set_array([])
    fcolors_shear_max = scalarmap_shear_max.to_rgba(stiffness_matrix.data_shear_3d["max"])

    axes.plot_surface(data_xyz_shear_min[0,:,:], data_xyz_shear_min[1,:,:], data_xyz_shear_min[2,:,:], facecolors=fcolors_shear_min, norm=norm_min, cmap="Greens", linewidth=0.05, edgecolor = 'k', alpha=0.8)
    axes.plot_surface(data_xyz_shear_max[0,:,:], data_xyz_shear_max[1,:,:], data_xyz_shear_max[2,:,:], facecolors=fcolors_shear_max, norm=norm_max, cmap="Blues", linewidth=0.05, edgecolor = 'k',
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


def plot_poisson_2d(stiffness_matrix: AbstractStiffnessTensor) -> None:
    """2D plotter for Poisson coefficient"""

    n_points = 100

    theta_array = np.linspace(0.0, np.pi, n_points)

    data_x_xy_1, data_y_xy_1 = make_planar_plot_data(np.array([poisson[0] * np.cos(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xy"], theta_array)]), np.array([poisson[0] * np.sin(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xy"], theta_array)]))
    data_x_xy_2, data_y_xy_2 = make_planar_plot_data(np.array([poisson[1] * np.cos(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xy"], theta_array)]), np.array([poisson[1] * np.sin(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xy"], theta_array)]))
    data_x_xy_3, data_y_xy_3 = make_planar_plot_data(np.array([poisson[2] * np.cos(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xy"], theta_array)]), np.array([poisson[2] * np.sin(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xy"], theta_array)]))

    data_x_xz_1, data_y_xz_1 = make_planar_plot_data(np.array([poisson[0] * np.sin(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xz"], theta_array)]), np.array([poisson[0] * np.cos(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xz"], theta_array)]))
    data_x_xz_2, data_y_xz_2 = make_planar_plot_data(np.array([poisson[1] * np.sin(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xz"], theta_array)]), np.array([poisson[1] * np.cos(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xz"], theta_array)]))
    data_x_xz_3, data_y_xz_3 = make_planar_plot_data(np.array([poisson[2] * np.sin(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xz"], theta_array)]), np.array([poisson[2] * np.cos(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["xz"], theta_array)]))

    data_x_yz_1, data_y_yz_1 = make_planar_plot_data(np.array([poisson[0] * np.sin(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["yz"], theta_array)]), np.array([poisson[0] * np.cos(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["yz"], theta_array)]))
    data_x_yz_2, data_y_yz_2 = make_planar_plot_data(np.array([poisson[1] * np.sin(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["yz"], theta_array)]), np.array([poisson[1] * np.cos(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["yz"], theta_array)]))
    data_x_yz_3, data_y_yz_3 = make_planar_plot_data(np.array([poisson[2] * np.sin(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["yz"], theta_array)]), np.array([poisson[2] * np.cos(theta) for poisson, theta in zip(stiffness_matrix.data_poisson_2d["yz"], theta_array)]))

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


def plot_poisson_3d(stiffness_matrix: AbstractStiffnessTensor) -> None:
    """3D plotter for Poisson coefficient"""

    n_points = 50

    theta_array = np.linspace(0.0, np.pi, n_points)
    phi_array = np.linspace(0.0, np.pi, n_points)
    phi_plus_pi_array = [phi_array[i] + np.pi for i in range(1, len(phi_array))]
    phi_array = np.append(phi_array, phi_plus_pi_array)

    data_xyz_poisson_1 = np.zeros((3, len(theta_array), len(phi_array)))
    data_xyz_poisson_2 = np.zeros((3, len(theta_array), len(phi_array)))
    data_xyz_poisson_3 = np.zeros((3, len(theta_array), len(phi_array)))

    for index_theta in range(len(theta_array)):
        for index_phi in range(len(phi_array)):
            z = np.cos(theta_array[index_theta])
            x = np.sin(theta_array[index_theta]) * np.cos(phi_array[index_phi])
            y = np.sin(theta_array[index_theta]) * np.sin(phi_array[index_phi])

            poisson_1 = stiffness_matrix.data_poisson_3d[0, index_theta, index_phi]
            data_xyz_poisson_1[0, index_theta, index_phi] = poisson_1 * x
            data_xyz_poisson_1[1, index_theta, index_phi] = poisson_1 * y
            data_xyz_poisson_1[2, index_theta, index_phi] = poisson_1 * z

            poisson_2 = stiffness_matrix.data_poisson_3d[1, index_theta, index_phi]
            data_xyz_poisson_2[0, index_theta, index_phi] = poisson_2 * x
            data_xyz_poisson_2[1, index_theta, index_phi] = poisson_2 * y
            data_xyz_poisson_2[2, index_theta, index_phi] = poisson_2 * z

            poisson_3 = stiffness_matrix.data_poisson_3d[2, index_theta, index_phi]
            data_xyz_poisson_3[0, index_theta, index_phi] = poisson_3 * x
            data_xyz_poisson_3[1, index_theta, index_phi] = poisson_3 * y
            data_xyz_poisson_3[2, index_theta, index_phi] = poisson_3 * z

    poisson_1_average = np.average(stiffness_matrix.data_poisson_3d[0, :, :])
    poisson_1_min = np.min(stiffness_matrix.data_poisson_3d[0, :, :])
    poisson_1_max = np.max(stiffness_matrix.data_poisson_3d[0, :, :])

    poisson_2_average = np.average(stiffness_matrix.data_poisson_3d[1, :, :])
    poisson_2_min = np.min(stiffness_matrix.data_poisson_3d[1, :, :])
    poisson_2_max = np.max(stiffness_matrix.data_poisson_3d[1, :, :])

    poisson_3_average = np.average(stiffness_matrix.data_poisson_3d[2, :, :])
    poisson_3_min = np.min(stiffness_matrix.data_poisson_3d[2, :, :])
    poisson_3_max = np.max(stiffness_matrix.data_poisson_3d[2, :, :])

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
    fcolors_poisson_1 = scalarmap_poisson_1.to_rgba(stiffness_matrix.data_poisson_3d[0, :, :])

    scalarmap_poisson_2 = cm.ScalarMappable(cmap="Greens", norm=norm_poisson_2)
    scalarmap_poisson_2.set_clim(poisson_2_min, poisson_2_max)
    scalarmap_poisson_2.set_array([])
    fcolors_poisson_2 = scalarmap_poisson_2.to_rgba(stiffness_matrix.data_poisson_3d[1, :, :])

    scalarmap_poisson_3 = cm.ScalarMappable(cmap="Blues", norm=norm_poisson_3)
    scalarmap_poisson_3.set_clim(poisson_3_min, poisson_3_max)
    scalarmap_poisson_3.set_array([])
    fcolors_poisson_3 = scalarmap_poisson_3.to_rgba(stiffness_matrix.data_poisson_3d[2, :, :])

    axes.plot_surface(data_xyz_poisson_1[0, :, :], data_xyz_poisson_1[1, :, :], data_xyz_poisson_1[2, :, :], facecolors=fcolors_poisson_1, norm=norm_poisson_1,
                      cmap="Reds", linewidth=0.1, edgecolor = 'k')
    axes.plot_surface(data_xyz_poisson_2[0, :, :], data_xyz_poisson_2[1, :, :], data_xyz_poisson_2[2, :, :], facecolors=fcolors_poisson_2, norm=norm_poisson_2,
                      cmap="Greens",
                      alpha=0.5, linewidth=0.1, edgecolor = 'k')
    axes.plot_surface(data_xyz_poisson_3[0, :, :], data_xyz_poisson_3[1, :, :], data_xyz_poisson_3[2, :, :], facecolors=fcolors_poisson_3, norm=norm_poisson_3, cmap="Blues",
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
