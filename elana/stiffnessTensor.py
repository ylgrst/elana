import numpy as np
import numpy.typing as npt
import math as m


class StiffnessTensor:
    """
    Class to manage stiffness tensors in Voigt notation (6x6 matrix)
    """

    def __init__(self,
                 matrix: npt.NDArray[np.float_] = np.zeros((6, 6))) -> None:
        self.matrix = matrix

    def read_matrix_from_txt(self, matrix_file : str) -> None:
        matrix: npt.NDArray[np.float_] = np.loadtxt(matrix_file)

        self.matrix = matrix

