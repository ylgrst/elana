import numpy as np
import pytest
from elana import AnisotropicStiffnessTensor


def rebuild_6x6_orthotropic_stiffness_tensor_from_properties(properties_array):
    """Rebuilds 6x6 orthotropic stiffness tensor from properties computed by ansys"""

    E1 = properties_array[0]/1e6
    E2 = properties_array[1]/1e6
    E3 = properties_array[2]/1e6
    G12 = properties_array[3]/1e6
    G23 = properties_array[4]/1e6
    G31 = properties_array[5]/1e6
    nu12 = properties_array[6]
    nu13 = properties_array[7]
    nu23 = properties_array[8]

    compliance = np.zeros((6,6))
    compliance[0, 0] = 1.0/E1
    compliance[0, 1] = -nu12/E2
    compliance[0, 2] = -nu13/E3
    compliance[1, 0] = compliance[0, 1]
    compliance[1, 1] = 1.0/E2
    compliance[1, 2] = -nu23/E3
    compliance[2, 0] = compliance[0, 2]
    compliance[2, 1] = compliance[1, 2]
    compliance[2, 2] = 1/E3
    compliance[3, 3] = 1/G23
    compliance[4, 4] = 1/G31
    compliance[5, 5] = 1/G12

    stiffness = np.round(np.linalg.inv(compliance), decimals=2)

    return stiffness

@pytest.fixture(scope='session')
def reference_properties():
    E1 = 6.9714E+07
    E2 = 6.9714E+07
    E3 = 6.9713E+07
    G12 = 4.3154E+07
    G23 = 4.3154E+07
    G31 = 4.3154E+07
    nu12 = 0.3131
    nu13 = 0.3131
    nu23 = 0.3131

    return np.array([E1, E2, E3, G12, G23, G31, nu12, nu13, nu23])


@pytest.fixture(scope='session')
def reference_stiffness_tensor(reference_properties):
    stiffness_tensor = rebuild_6x6_orthotropic_stiffness_tensor_from_properties(reference_properties)
    return stiffness_tensor


def test_compute_xyz_properties_from_reference_stiffness_tensor_must_return_same_array_as_reference_properties_array(reference_properties, reference_stiffness_tensor):
    stiff = AnisotropicStiffnessTensor(reference_stiffness_tensor)
    E_xyz = stiff.young_xyz()
    G_xyz = stiff.shear_xyz()
    nu_xyz = stiff.poisson_xyz()

    properties = np.array([E_xyz[0]*1e6, E_xyz[1]*1e6, E_xyz[2]*1e6, G_xyz[0]*1e6, G_xyz[1]*1e6, G_xyz[2]*1e6, nu_xyz[0], nu_xyz[2], nu_xyz[1]])

    assert np.all(np.isclose(properties, reference_properties, rtol=1e-3))