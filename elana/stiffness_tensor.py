import numpy as np
import numpy.typing as npt
from .operations import compute_direction_vector_spherical_to_cartesian
from scipy import optimize

_VOIGT_MATRIX: npt.NDArray[int] = np.array([[0, 5, 4], [5, 1, 3], [4, 3, 2]])


def _compute_voigt_coefficient(p: int, q: int) -> float:
    return 1. / ((1 + p // 3) * (1 + q // 3))


class StiffnessTensor:
    """
    Class to manage stiffness tensors (6x6 matrix)
    """

    def __init__(self,
                 matrix: npt.NDArray[np.float_] = np.zeros((6, 6)),
                 flexibility_matrix: npt.NDArray[np.float_] = np.zeros((6, 6)),
                 flexibility_tensor: npt.NDArray[np.float_] = np.zeros((3, 3, 3, 3))) -> None:
        self.matrix = matrix
        self.flexibility_matrix = flexibility_matrix
        self.flexibility_tensor = flexibility_tensor

    def read_matrix_from_txt(self, matrix_file: str) -> None:
        matrix: npt.NDArray[np.float_] = np.loadtxt(matrix_file)

        if matrix.shape != (6, 6):
            raise ValueError("Matrix shape should be 6x6")
        if np.linalg.norm(matrix - matrix.transpose()) > 1e-3:
            raise ValueError("Matrix should be symmetric")

        self.matrix = matrix

        try:
            self.flexibility_matrix = np.linalg.inv(matrix)
        except:
            raise ValueError("Matrix is singular")

    def _convert_to_tensor(self) -> npt.NDArray[np.float_]:
        self.flexibility_tensor = [[[[_compute_voigt_coefficient(_VOIGT_MATRIX[i][j], _VOIGT_MATRIX[k][l]) *
                                      self.flexibility_matrix[_VOIGT_MATRIX[i][j]][_VOIGT_MATRIX[k][l]]
                                      for i in range(3)] for j in range(3)] for k in range(3)] for l in range(3)]

    def is_orthorhombic(self) -> bool:
        stiffness_matrix_orthorhombic_coefficient_list: list[float] = [self.matrix[0][3], self.matrix[0][4],
                                                                       self.matrix[0][5], self.matrix[1][3],
                                                                       self.matrix[1][4], self.matrix[1][5],
                                                                       self.matrix[2][3], self.matrix[2][4],
                                                                       self.matrix[2][5], self.matrix[3][4],
                                                                       self.matrix[3][5], self.matrix[4][5]]

        return np.isclose(stiffness_matrix_orthorhombic_coefficient_list, [0.0] * 12)

    def is_cubic(self) -> bool:
        stiffness_matrix_cubic_coefficient_list: list[float] = [self.matrix[0][3], self.matrix[0][4], self.matrix[0][5],
                                                                self.matrix[1][3], self.matrix[1][4], self.matrix[1][5],
                                                                self.matrix[2][3], self.matrix[2][4], self.matrix[2][5],
                                                                self.matrix[3][4], self.matrix[3][5], self.matrix[4][5],
                                                                self.matrix[0][0] - self.matrix[1][1],
                                                                self.matrix[0][0] - self.matrix[2][2],
                                                                self.matrix[3][3] - self.matrix[4][4],
                                                                self.matrix[3][3] - self.matrix[5][5],
                                                                self.matrix[0][1] - self.matrix[0][2],
                                                                self.matrix[0][1] - self.matrix[1][2]]

        return np.isclose(stiffness_matrix_cubic_coefficient_list, [0.0] * 18)

    def young(self, angles: tuple[float, float]) -> float:
        direction_vector: list[float] = compute_direction_vector_spherical_to_cartesian(angles[0], angles[1])
        result: float = sum([
            direction_vector[i] * direction_vector[j] * direction_vector[k] * direction_vector[l] *
            self.flexibility_tensor[i][j][k][l]
            for i in range(3) for j in range(3) for k in range(3) for l in range(3)
        ])
        return 1 / result

    def linear_compressibility(self, angles: tuple[float, float]) -> float:
        direction_vector: list[float] = compute_direction_vector_spherical_to_cartesian(angles[0], angles[1])
        result: float = sum([direction_vector[i] * direction_vector[j] * self.flexibility_tensor[i][j][k][k]
                             for i in range(3) for j in range(3) for k in range(3)])
        return 1000 * result

    def shear(self, angles: tuple[float, float, float]) -> float:
        direction_vector_1 = compute_direction_vector_spherical_to_cartesian(angles[0], angles[1])
        direction_vector_2 = compute_direction_vector_spherical_to_cartesian(angles[0], angles[1], angles[2])
        result = sum([direction_vector_1[i] * direction_vector_2[j] * direction_vector_1[k] * direction_vector_2[l] *
                      self.flexibility_tensor[i][j][k][l]
                      for i in range(3) for j in range(3) for k in range(3) for l in range(3)])
        return 1 / (4 * result)

    def poisson(self, angles: tuple[float, float, float]) -> float:
        direction_vector_1 = compute_direction_vector_spherical_to_cartesian(angles[0], angles[1])
        direction_vector_2 = compute_direction_vector_spherical_to_cartesian(angles[0], angles[1], angles[2])

        result_numerator = - sum([direction_vector_1[i] * direction_vector_1[j] * direction_vector_2[k] *
                                  direction_vector_2[l] * self.flexibility_tensor[i][j][k][l] for i in range(3) for j in
                                  range(3) for k in range(3) for l in range(3)])
        result_denominator = sum([direction_vector_1[i] * direction_vector_1[j] * direction_vector_1[k] *
                                  direction_vector_1[l] * self.flexibility_tensor[i][j][k][l] for i in range(3) for j in
                                  range(3) for k in range(3) for l in range(3)])

        return result_numerator / result_denominator

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
        tmpA = (self.flexibility_matrix[0][0] + self.flexibility_matrix[1][1] + self.flexibility_matrix[2][2]) / 3
        tmpB = (self.flexibility_matrix[1][2] + self.flexibility_matrix[0][2] + self.flexibility_matrix[0][1]) / 3
        tmpC = (self.flexibility_matrix[3][3] + self.flexibility_matrix[4][4] + self.flexibility_matrix[5][5]) / 3

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
