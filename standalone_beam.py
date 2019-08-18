from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import minimize


def assemble_CSC_K(K_local):
    """
    Assemble the stiffness matrix in sparse CSC format.

    Returns
    -------
    ndarray
        Stiffness matrix as dense ndarray.
    """
    num_nodes = num_elements + 1
    num_entry = num_elements * 12 + 4
    ndim = num_entry + 4

    data = np.zeros((ndim, ))
    cols = np.empty((ndim, ))
    rows = np.empty((ndim, ))

    # First element.
    data[:16] = K_local[0, :, :].flat
    cols[:16] = np.tile(np.arange(4), 4)
    rows[:16] = np.repeat(np.arange(4), 4)

    j = 16
    for ind in range(1, num_elements):
        ind1 = 2 * ind
        K = K_local[ind, :, :]

        # NW quadrant gets summed with previous connected element.
        data[j-6:j-4] += K[0, :2]
        data[j-2:j] += K[1, :2]

        # NE quadrant
        data[j:j+4] = K[:2, 2:].flat
        rows[j:j+4] = np.array([ind1, ind1, ind1 + 1, ind1 + 1])
        cols[j:j+4] = np.array([ind1 + 2, ind1 + 3, ind1 + 2, ind1 + 3])

        # SE and SW quadrants together
        data[j+4:j+12] = K[2:, :].flat
        rows[j+4:j+12] = np.repeat(np.arange(ind1 + 2, ind1 + 4), 4)
        cols[j+4:j+12] = np.tile(np.arange(ind1, ind1 + 4), 2)

        j += 12

    data[-4:] = 1.0
    rows[-4] = 2 * num_nodes
    rows[-3] = 2 * num_nodes + 1
    rows[-2] = 0.0
    rows[-1] = 1.0
    cols[-4] = 0.0
    cols[-3] = 1.0
    cols[-2] = 2 * num_nodes
    cols[-1] = 2 * num_nodes + 1

    n_K = 2 * num_nodes + 2
    return coo_matrix((data, (rows, cols)), shape=(n_K, n_K)).tocsc()



def beam_model(h, E=1., L=1., b=0.1, num_elements=50):
    num_nodes = num_elements + 1

    force_vector = np.zeros(2 * num_nodes)
    force_vector[-2] = -1.

    # I_comp = MomentOfInertiaComp(num_elements=num_elements, b=b)
    I = 1./12. * b * h ** 3


    # comp = LocalStiffnessMatrixComp(num_elements=num_elements, E=E, L=L)
    L0 = L / num_elements
    coeffs = np.empty((4, 4))
    coeffs[0, :] = [12, 6 * L0, -12, 6 * L0]
    coeffs[1, :] = [6 * L0, 4 * L0 ** 2, -6 * L0, 2 * L0 ** 2]
    coeffs[2, :] = [-12, -6 * L0, 12, -6 * L0]
    coeffs[3, :] = [6 * L0, 2 * L0 ** 2, -6 * L0, 4 * L0 ** 2]
    coeffs *= E / L0 ** 3

    mtx = np.zeros((num_elements, 4, 4, num_elements))
    for ind in range(num_elements):
        mtx[ind, :, :, ind] = coeffs

    K_local = np.zeros((num_elements, 4, 4))
    for ind in range(num_elements):
        K_local[ind, :, :] = mtx[ind, :, :, ind] * I[ind]

    # comp = StatesComp(num_elements=num_elements, force_vector=force_vector)
    force_vector = np.concatenate([force_vector, np.zeros(2)])

    K = assemble_CSC_K(K_local)
    lu = splu(K)

    d = lu.solve(force_vector)

    displacements = d[2 * num_nodes]

    # comp = ComplianceComp(num_elements=num_elements, force_vector=force_vector)
    compliance = np.dot(force_vector, displacements)
    
    return compliance

    # self.add_design_var('inputs_comp.h', lower=1e-2, upper=10.)
    # self.add_objective('compliance_comp.compliance')
    
def volume_function(h, L, b, num_elements):
    L0 = L / num_elements

    volume = np.sum(h * b * L0)

num_elements=50
E=1.
L=1.
b=0.1
volume=0.01
num_elements=50
h0 = np.ones((num_elements))

constraint_dict = {
    'type' : 'eq',
    'fun' : volume_function,
    'args' : [h, L, b, num_elements]
}

result = minimize(beam_model, h0, args=[E, L, b, num_elements], constraints=constraint_dict)

print(result.x)
