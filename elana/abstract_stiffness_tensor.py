from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np
from scipy import optimize
from elana.operations import _compute_4th_order_tensor_from_6x6_matrix

class AbstractStiffnessTensor(ABC):
    def __init__(self,
                 matrix: npt.NDArray[np.float_]) -> None:

        if matrix.shape != (6, 6):
            raise ValueError("Matrix shape should be 6x6")
        if np.linalg.norm(matrix - matrix.transpose()) > 1e-3:
            raise ValueError("Matrix should be symmetric")

        self.matrix = matrix

        try:
            self.compliance_matrix = np.linalg.inv(matrix)
        except:
            raise ValueError("Matrix is singular")

        self.compliance_tensor = _compute_4th_order_tensor_from_6x6_matrix(self.compliance_matrix)

        self._build_young_2d_plot_data()
        self._build_young_3d_plot_data()
        self._build_linear_compressibility_2d_plot_data()
        self._build_linear_compressibility_3d_plot_data()


    @abstractmethod
    def young(self, angles: tuple[float, float]) -> float:
        pass

    @abstractmethod
    def linear_compressibility(self, angles: tuple[float, float]) -> float:
        pass

    @abstractmethod
    def shear(self, angles: tuple[float, float, float]) -> float:
        pass

    @abstractmethod
    def poisson(self, angles: tuple[float, float, float]) -> float:
        pass

    def voigt_averages(self) -> list[float]:
        tmpA = (self.matrix[0][0] + self.matrix[1][1] + self.matrix[2][2]) / 3
        tmpB = (self.matrix[1][2] + self.matrix[0][2] + self.matrix[0][1]) / 3
        tmpC = (self.matrix[3][3] + self.matrix[4][4] + self.matrix[5][5]) / 3

        bulk_modulus = (tmpA + 2 * tmpB) / 3
        shear_modulus = (tmpA - tmpB + 3 * tmpC) / 5
        young_modulus = 1 / (1 / (3 * shear_modulus) + 1 / (9 * bulk_modulus))
        poisson_ratio = (1 - 3 * shear_modulus / (3 * bulk_modulus + shear_modulus)) / 2

        return [bulk_modulus, shear_modulus, young_modulus, poisson_ratio]

    def reuss_averages(self) -> list[float]:
        tmpA = (self.compliance_matrix[0][0] + self.compliance_matrix[1][1] + self.compliance_matrix[2][2]) / 3
        tmpB = (self.compliance_matrix[1][2] + self.compliance_matrix[0][2] + self.compliance_matrix[0][1]) / 3
        tmpC = (self.compliance_matrix[3][3] + self.compliance_matrix[4][4] + self.compliance_matrix[5][5]) / 3

        bulk_modulus = 1 / (3 * tmpA + 6 * tmpB)
        shear_modulus = 5 / (4 * tmpA - 4 * tmpB + 3 * tmpC)
        young_modulus = 1 / (1 / (3 * shear_modulus) + 1 / (9 * bulk_modulus))
        poisson_ratio = (1 - 3 * shear_modulus / (3 * bulk_modulus + shear_modulus)) / 2

        return [bulk_modulus, shear_modulus, young_modulus, poisson_ratio]

    def hill_averages(self) -> list[float]:
        bulk_modulus = (self.voigt_averages()[0] + self.reuss_averages()[0]) / 2
        shear_modulus = (self.voigt_averages()[1] + self.reuss_averages()[1]) / 2
        young_modulus = 1 / (1 / (3 * shear_modulus) + 1 / (9 * bulk_modulus))
        poisson_ratio = (1 - 3 * shear_modulus / (3 * bulk_modulus + shear_modulus)) / 2

        return [bulk_modulus, shear_modulus, young_modulus, poisson_ratio]


    def shear_2d(self, angles: tuple[float, float]) -> tuple[float, float]:
        ftol = 0.001
        xtol = 0.01

        def shear_function(z): return self.shear((angles[0], angles[1], z))

        def minus_shear_function(z): return -self.shear((angles[0], angles[1], z))

        result_pos = optimize.minimize(shear_function, np.pi / 2.0, args=(), method='Powell',
                                       options={"xtol": xtol, "ftol": ftol})  # , bounds=[(0.0,np.pi)])
        result_neg = optimize.minimize(minus_shear_function, np.pi / 2.0, args=(), method='Powell',
                                       options={"xtol": xtol, "ftol": ftol})  # , bounds=[(0.0,np.pi)])
        return float(result_pos.fun), -float(result_neg.fun)

    def shear_3d(self, angles: tuple[float, float]) -> tuple[float, float, float, float]:
        tol = 0.005
        guess = np.pi / 2.0

        def shear_function(z): return self.shear((angles[0], angles[1], z))

        def minus_shear_function(z): return -self.shear((angles[0], angles[1], z))

        result_pos = optimize.minimize(shear_function, guess, args=(), method='COBYLA',
                                       options={"tol": tol})  # , bounds=[(0.0,np.pi)])
        result_neg = optimize.minimize(minus_shear_function, guess, args=(), method='COBYLA',
                                       options={"tol": tol})  # , bounds=[(0.0,np.pi)])

        return float(result_pos.fun), -float(result_neg.fun), float(result_pos.x), float(result_neg.x)

    def poisson_2d(self, angles: tuple[float, float]) -> tuple[float, float, float]:
        ftol = 0.001
        xtol = 0.01

        def poisson_function(z): return self.poisson((angles[0], angles[1], z))

        def minus_poisson_function(z): return -self.poisson((angles[0], angles[1], z))

        result_pos = optimize.minimize(poisson_function, np.pi / 2.0, args=(), method='Powell',
                                       options={"xtol": xtol, "ftol": ftol})  # , bounds=[(0.0,np.pi)])
        result_neg = optimize.minimize(minus_poisson_function, np.pi / 2.0, args=(), method='Powell',
                                       options={"xtol": xtol, "ftol": ftol})  # , bounds=[(0.0,np.pi)])
        return min(0.0, float(result_pos.fun)), max(0.0, float(result_pos.fun)), -float(result_neg.fun)

    def poisson_3d(self, angles: tuple[float, float]) -> tuple[float, float, float, float, float]:
        tol = 0.005
        guess = np.pi / 2.0

        def poisson_function(z): return self.poisson((angles[0], angles[1], z))

        def minus_poisson_function(z): return -self.poisson((angles[0], angles[1], z))

        result_pos = optimize.minimize(poisson_function, guess, args=(), method='COBYLA',
                                       options={"tol": tol})  # , bounds=[(0.0,np.pi)])
        result_neg = optimize.minimize(minus_poisson_function, guess, args=(), method='COBYLA',
                                       options={"tol": tol})  # , bounds=[(0.0,np.pi)])

        return min(0.0, float(result_pos.fun)), max(0.0, float(result_pos.fun)), -float(result_neg.fun), float(
            result_pos.x), float(result_neg.x)

    def _build_young_2d_plot_data(self) -> None:
        n_points = 100

        theta_array = np.linspace(0.0, np.pi, n_points)

        young_xy = list(map(lambda x: self.young((np.pi / 2.0, x)), theta_array))
        young_xz = list(map(lambda x: self.young((x, 0.0)), theta_array))
        young_yz = list(map(lambda x: self.young((x, np.pi / 2.0)), theta_array))

        data_young_2d = {"xy": young_xy, "xz": young_xz, "yz": young_yz}

        self.data_young_2d = data_young_2d

    def _build_young_3d_plot_data(self) -> None:
        n_points = 200

        theta_array = np.linspace(0.0, np.pi, n_points)
        phi_array = np.linspace(0.0, 2 * np.pi, 2 * n_points)

        data_young = np.zeros((n_points, 2 * n_points))

        for index_theta in range(len(theta_array)):
            for index_phi in range(len(phi_array)):
                young = self.young((theta_array[index_theta], phi_array[index_phi]))
                data_young[index_theta, index_phi] = young


        self.data_young_3d = data_young

    def _build_linear_compressibility_2d_plot_data(self) -> None:
        n_points = 100

        theta_array = np.linspace(0.0, np.pi, n_points)

        linear_compressibility_pos_xy = list(
            map(lambda x: max(0.0, self.linear_compressibility((np.pi / 2.0, x))), theta_array))
        linear_compressibility_pos_xz = list(
            map(lambda x: max(0.0, self.linear_compressibility((x, 0.0))), theta_array))
        linear_compressibility_pos_yz = list(
            map(lambda x: max(0.0, self.linear_compressibility((x, np.pi / 2.0))), theta_array))

        linear_compressibility_neg_xy = list(
            map(lambda x: max(0.0, -self.linear_compressibility((np.pi / 2.0, x))), theta_array))
        linear_compressibility_neg_xz = list(
            map(lambda x: max(0.0, -self.linear_compressibility((x, 0.0))), theta_array))
        linear_compressibility_neg_yz = list(
            map(lambda x: max(0.0, -self.linear_compressibility((x, np.pi / 2.0))), theta_array))

        data_linear_compressibility_2d_pos = {"xy": linear_compressibility_pos_xy, "xz": linear_compressibility_pos_xz, "yz": linear_compressibility_pos_yz}
        data_linear_compressibility_2d_neg = {"xy": linear_compressibility_neg_xy, "xz": linear_compressibility_neg_xz, "yz": linear_compressibility_neg_yz}
        data_linear_compressibility_2d = {"pos" : data_linear_compressibility_2d_pos, "neg": data_linear_compressibility_2d_neg}

        self.data_linear_compressibility_2d = data_linear_compressibility_2d

    def _build_linear_compressibility_3d_plot_data(self) -> None:
        n_points = 200

        theta_array = np.linspace(0.0, np.pi, n_points)
        phi_array = np.linspace(0.0, 2 * np.pi, 2 * n_points)

        data_linear_compressibility_pos = np.zeros((n_points, 2 * n_points))
        data_linear_compressibility_neg = np.zeros((n_points, 2 * n_points))

        for index_theta in range(len(theta_array)):
            for index_phi in range(len(phi_array)):
                linear_compressibility_pos = max(0.0, self.linear_compressibility(
                    (theta_array[index_theta], phi_array[index_phi])))
                linear_compressibility_neg = max(0.0, -self.linear_compressibility(
                    (theta_array[index_theta], phi_array[index_phi])))

                data_linear_compressibility_pos[index_theta, index_phi] = linear_compressibility_pos
                data_linear_compressibility_neg[index_theta, index_phi] = linear_compressibility_neg

        data_linear_compressibility_3d = {"pos": data_linear_compressibility_pos, "neg": data_linear_compressibility_neg}

        self.data_linear_compressibility_3d = data_linear_compressibility_3d




