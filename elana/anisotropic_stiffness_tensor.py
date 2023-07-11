import numpy as np
import numpy.typing as npt
from .operations import compute_direction_vector_spherical_to_cartesian
from abstract_stiffness_tensor import AbstractStiffnessTensor


class AnisotropicStiffnessTensor(AbstractStiffnessTensor):
    """
    Class to manage anisotropic stiffness tensors
    """

    def __init__(self,
                 matrix: npt.NDArray[np.float_] = np.zeros((6, 6))) -> None:
        super().__init__(matrix)

#TODO add subclass Young, shear, LC, poisson (other file) and add corresponding attribute to StiffnessTensor, + add constructors to build data, separate plots from data construction

    def young(self, angles: tuple[float, float]) -> float:
        direction_vector: list[float] = compute_direction_vector_spherical_to_cartesian(angles[0], angles[1])
        result: float = sum([
            direction_vector[i] * direction_vector[j] * direction_vector[k] * direction_vector[l] *
            self.flexibility_tensor[i][j][k][l]
            for i in range(3) for j in range(3) for k in range(3) for l in range(3)
        ])
        return 1 / result

#TODO Implement min, max, anisotropy for young, lc, shear, poisson
#TODO Implement orthogonal check on vector list (scalar pdct)

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


