import math as m
from .stiffness_tensor import StiffnessTensor


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

    def shear(self, angles: tuple[float, float, float]) -> float:
        cos_theta = m.cos(angles[0])
        cos_theta_squared = cos_theta*cos_theta
        sin_theta_squared = 1 - cos_theta_squared
        cos_phi = m.cos(angles[1])
        sin_phi = m.sin(angles[1])
        sin_phi_squared = sin_phi * sin_phi
        cos_chi = m.cos(angles[2])
        cos_chi_squared = cos_chi * cos_chi
        sin_chi = m.sin(angles[2])
        sin_chi_squared = 1 - cos_chi_squared

        s11 = self.flexibility_tensor[0][0][0][0]
        s22 = self.flexibility_tensor[1][1][1][1]
        s33 = self.flexibility_tensor[2][2][2][2]
        s44 = 4 * self.flexibility_tensor[1][2][1][2]
        s55 = 4 * self.flexibility_tensor[0][2][0][2]
        s66 = 4 * self.flexibility_tensor[0][1][0][1]
        s12 = self.flexibility_tensor[0][0][1][1]
        s13 = self.flexibility_tensor[0][0][2][2]
        s23 = self.flexibility_tensor[1][1][2][2]

        result = (
            cos_theta_squared*cos_theta_squared*cos_chi_squared*s44*sin_phi_squared + cos_chi_squared*s44*sin_phi_squared*sin_theta_squared*sin_theta_squared + 4*cos_phi**3*cos_theta*cos_chi*(-2*s11 + 2*s12 + s66)*sin_phi*sin_theta_squared*sin_chi
            + 2*cos_phi*cos_theta*cos_chi*sin_phi*(cos_theta_squared*(s44 - s55) + (4*s13 - 4*s23 - s44 + s55 - 4*s12*sin_phi_squared + 4*s22*sin_phi_squared - 2*s66*sin_phi_squared)*sin_theta_squared)*sin_chi
            + s66*sin_phi_squared*sin_phi_squared*sin_theta_squared*sin_chi_squared + cos_phi**4*sin_theta_squared*(4*cos_theta_squared*cos_chi_squared*s11 + s66*sin_chi_squared)
            + cos_theta_squared*(2*cos_chi_squared*(2*s33 + sin_phi_squared*(-4*s23 - s44 + 2*s22*sin_phi_squared))*sin_theta_squared + s55*sin_phi_squared*sin_chi_squared)
            + cos_phi**2*(cos_theta_squared*cos_theta_squared*cos_chi_squared*s55 + cos_theta_squared*(-2*cos_chi_squared*(4*s13 + s55 - 2*(2*s12 + s66)*sin_phi_squared)*sin_theta_squared + s44*sin_chi_squared)
                     + sin_theta_squared*(cos_chi_squared*s55*sin_theta_squared + 2*(2*s11 - 4*s12 + 2*s22 - s66)*sin_phi_squared*sin_chi_squared))
            )

        return 1/result

    def poisson(self, angles: tuple[float, float, float]) -> float:
        cos_theta = m.cos(angles[0])
        sin_theta_squared = m.sin(angles[0]) * m.sin(angles[0])
        cos_phi = m.cos(angles[1])
        sin_phi = m.sin(angles[1])
        cos_chi = m.cos(angles[2])
        sin_chi = m.sin(angles[2])

        s11 = self.flexibility_tensor[0][0][0][0]
        s22 = self.flexibility_tensor[1][1][1][1]
        s33 = self.flexibility_tensor[2][2][2][2]
        s44 = 4 * self.flexibility_tensor[1][2][1][2]
        s55 = 4 * self.flexibility_tensor[0][2][0][2]
        s66 = 4 * self.flexibility_tensor[0][1][0][1]
        s12 = self.flexibility_tensor[0][0][1][1]
        s13 = self.flexibility_tensor[0][0][2][2]
        s23 = self.flexibility_tensor[1][1][2][2]

        result =  (-(cos_theta**2*cos_chi**2*s33*sin_theta_squared) - cos_phi**2*cos_chi**2*s13*sin_theta_squared*sin_theta_squared - cos_chi**2*s23*sin_phi**2*sin_theta_squared*sin_theta_squared + cos_theta*cos_chi*s44*sin_phi*sin_theta_squared*(cos_theta*cos_chi*sin_phi + cos_phi*sin_chi) -
            cos_theta**2*s23*(cos_theta*cos_chi*sin_phi + cos_phi*sin_chi)**2 - cos_phi**2*s12*sin_theta_squared*(cos_theta*cos_chi*sin_phi + cos_phi*sin_chi)**2 - s22*sin_phi**2*sin_theta_squared*(cos_theta*cos_chi*sin_phi + cos_phi*sin_chi)**2 +
            cos_phi*cos_theta*cos_chi*s55*sin_theta_squared*(cos_phi*cos_theta*cos_chi - sin_phi*sin_chi) - cos_phi*s66*sin_phi*sin_theta_squared*(cos_theta*cos_chi*sin_phi + cos_phi*sin_chi)*(cos_phi*cos_theta*cos_chi - sin_phi*sin_chi) -
            cos_theta**2*s13*(cos_phi*cos_theta*cos_chi - sin_phi*sin_chi)**2 - cos_phi**2*s11*sin_theta_squared*(cos_phi*cos_theta*cos_chi - sin_phi*sin_chi)**2 - s12*sin_phi**2*sin_theta_squared*(cos_phi*cos_theta*cos_chi - sin_phi*sin_chi)**2)/(cos_theta**4*s33 + 2*cos_phi**2*cos_theta**2*s13*sin_theta_squared + cos_phi**2*cos_theta**2*s55*sin_theta_squared + 2*cos_theta**2*s23*sin_phi**2*sin_theta_squared + cos_theta**2*s44*sin_phi**2*sin_theta_squared +
            cos_phi**4*s11*sin_theta_squared*sin_theta_squared + 2*cos_phi**2*s12*sin_phi**2*sin_theta_squared*sin_theta_squared + cos_phi**2*s66*sin_phi**2*sin_theta_squared*sin_theta_squared + s22*sin_phi**4*sin_theta_squared*sin_theta_squared)

        return result