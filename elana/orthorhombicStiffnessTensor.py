import math as m
from .stiffnessTensor import StiffnessTensor


class Orthorhombic(StiffnessTensor):
    """A stiffness tensor in the case of an orthorhombic system"""

    def __init__(self, stiffness_tensor_initiator):
        """Initialize from a matrix, or from an Elastic object"""
        if type(stiffness_tensor_initiator) == str:
            StiffnessTensor.__init__(self, stiffness_tensor_initiator)
        elif isinstance(stiffness_tensor_initiator, StiffnessTensor):
            self.matrix = stiffness_tensor_initiator.matrix
            self.flexibility_matrix = stiffness_tensor_initiator.flexibility_matrix
            self.flexibility_tensor = stiffness_tensor_initiator.flexibility_tensor
        else:
            raise TypeError("Orthorhombic constructor argument should be string or Elastic object")

    def young(self, angles: tuple[float, float]) -> float:

        cos_theta_squared = m.cos(angles[0]) ** 2
        sin_theta_squared = 1 - cos_theta_squared
        cos_phi_squared = m.cos(angles[1]) ** 2
        sin_phi_squared = 1 - cos_phi_squared
        s11 = self.flexibility_tensor[0][0][0][0]
        s22 = self.flexibility_tensor[1][1][1][1]
        s33 = self.flexibility_tensor[2][2][2][2]
        s44 = 4 * self.flexibility_tensor[1][2][1][2]
        s55 = 4 * self.flexibility_tensor[0][2][0][2]
        s66 = 4 * self.flexibility_tensor[0][1][0][1]
        s12 = self.flexibility_tensor[0][0][1][1]
        s13 = self.flexibility_tensor[0][0][2][2]
        s23 = self.flexibility_tensor[1][1][2][2]

        young_modulus = 1 / (
                    cos_theta_squared ** 2 * s33 + 2 * cos_phi_squared * cos_theta_squared * s13 * sin_theta_squared + cos_phi_squared * cos_theta_squared * s55 * sin_theta_squared + 2 * cos_theta_squared * s23 * sin_phi_squared * sin_theta_squared + cos_theta_squared * s44 * sin_phi_squared * sin_theta_squared + cos_phi_squared ** 2 * s11 * sin_theta_squared ** 2 + 2 * cos_phi_squared * s12 * sin_phi_squared * sin_theta_squared ** 2 + cos_phi_squared * s66 * sin_phi_squared * sin_theta_squared ** 2 + s22 * sin_phi_squared ** 2 * sin_theta_squared ** 2)

        return young_modulus

    def linear_compressibility(self, angles: tuple[float, float]) -> float:
        cos_theta_squared = m.cos(angles[0]) ** 2
        cos_phi_squared = m.cos(angles[1]) ** 2
        s11 = self.flexibility_tensor[0][0][0][0]
        s22 = self.flexibility_tensor[1][1][1][1]
        s33 = self.flexibility_tensor[2][2][2][2]
        s12 = self.flexibility_tensor[0][0][1][1]
        s13 = self.flexibility_tensor[0][0][2][2]
        s23 = self.flexibility_tensor[1][1][2][2]

        linear_compressibility_modulus = 1000 * (cos_theta_squared * (s13 + s23 + s33) + (cos_phi_squared * (s11 + s12 + s13) + (s12 + s22 + s23) * (1 - cos_phi_squared)) * (1 - cos_theta_squared))

        return linear_compressibility_modulus

    