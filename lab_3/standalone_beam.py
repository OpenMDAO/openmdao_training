from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import minimize, Bounds


def assemble_CSC_K(K_local, num_elements):
    """
    Assemble the stiffness matrix in sparse CSC format.
    This takes in the local stiffness matrices and assembles a full
    stiffness matrix of all the elements.

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


def assemble_K_local(h, E, L, b, num_elements): 
    # Compute moment of inertia
    I = 1./12. * b * h ** 3

    # Compute local stiffness matrices
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

    return K_local

def beam_model(h, E, L, b, num_elements):
    """
    This is the main function that evaluates the performance of a beam model.

    It takes in data for the beam, applies a load, computes the
    displacements, and returns the compliance of the structure.
    """
    num_nodes = num_elements + 1

    # Create force vector
    force_vector = np.zeros(2 * num_nodes)
    force_vector[-2] = -1.

    
    # Solve linear system to obtain displacements
    force_vector = np.concatenate([force_vector, np.zeros(2)])

    K_local = assemble_K_local(h, E, L, b, num_elements)
    K = assemble_CSC_K(K_local, num_elements)
    lu = splu(K)

    displacements = lu.solve(force_vector)


    return displacements, force_vector

def beam_FEM_residuals(h, E, L, b, num_elements, u):
    """given the inputs (h, E, L, b, num_elements) and 
       the state vector (u) return the residuals
    """

    num_nodes = num_elements + 1

    # Create force vector
    force_vector = np.zeros(2 * num_nodes)
    force_vector[-2] = -1.

    
    # Solve linear system to obtain displacements
    force_vector = np.concatenate([force_vector, np.zeros(2)])

    K_local = assemble_K_local(h, E, L, b, num_elements)
    K = assemble_CSC_K(K_local, num_elements)

    u_residuals = K.dot(u) - force_vector
    return u_residuals, force_vector


def compliance_function(force_vector, displacements):
    # Compute and return the compliance of the beam
    compliance = np.dot(force_vector, displacements)
    return compliance



def volume_function(h, L, b, num_elements):
    """
    This function computes the volume of a beam structure
    """

    L0 = L / num_elements
    return np.sum(h * b * L0)

def volume_constraint(h, L, b, num_elements, req_volume):
    """
    Computes the actual optimization constraint required by scipy. 
    This won't be used by the OpenMDAO wrapper.
    """
    vol = volume_function(h, L, b, num_elements)

    volume_diff = req_volume - volume_function(h, L, b, num_elements)

    return volume_diff

if __name__ == "__main__": 

    import sys

    if len(sys.argv) == 1: 
        sys.argv.append('solve')

    if sys.argv[1] == "solve": 
        print('solve call')

        ################################################
        # simple run script that reads inputs from 
        # input.txt and writes to output.txt
        ################################################

        import os
        import numpy as np

        from standalone_beam import volume_function, beam_model

        # this will pull all the inputs into the global namespace
        with open('input.txt', 'r') as f: 
            inp = f.read()
            exec(inp)
            # h, E, L, b, num_elements are now assigned

        u, force_vector = beam_model(h, E, L, b, num_elements)
        compliance = compliance_function(force_vector, u)
        volume = volume_function(h, L, b, num_elements)

        with open('output.txt', 'w') as f: 
            f.write('u = {}\n'.format(u.tolist()))
            f.write('compliance = {}\n'.format(compliance))
            f.write('volume = {}'.format(volume))

    elif sys.argv[1] == "apply": 
        print('apply call')

        # this will pull all the inputs into the global namespace
        with open('input.txt', 'r') as f: 
            inp = f.read()
            exec(inp)
            # h, E, L, b, num_elements, and u, c, v are now assigned

        u_residuals, force_vector = beam_FEM_residuals(h, E, L, b, num_elements, u)
        c_residual = compliance - compliance_function(force_vector, u)
        v_residual = volume - volume_function(h, L, b, num_elements)

        with open('output.txt', 'w') as f: 
            f.write('u_residuals = {}\n'.format(u_residuals.tolist()))
            f.write('c_residual = {}\n'.format(c_residual))
            f.write('v_residual = {}\n'.format(v_residual))


    if sys.argv[1] == "opt": 
        ##############################################
        #run an optimization using FD and scipy 
        ##############################################

        num_elements = 50
        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01
        h = np.ones((num_elements)) * 1.0

        constraint_dict = {
            'type' : 'eq',
            'fun' : volume_constraint,
            'args' : (L, b, num_elements, volume),
        }

        bounds = Bounds(0.01, 10.)
        result = minimize(beam_model, h, tol=1e-9, bounds=bounds, args=(E, L, b, num_elements), constraints=constraint_dict, options={'maxiter' : 500})

        print('Optimal element height distribution:')
        print(result.x)