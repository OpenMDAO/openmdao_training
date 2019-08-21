from __future__ import division
import numpy as np

import openmdao.api as om

# all of these components have already been created for you,
# but look in beam_comp.py if you're curious to see how
from beam_comps import (MomentOfInertiaComp, LocalStiffnessMatrixComp, FEM,
                        ComplianceComp, VolumeComp)


class BeamGroup(om.Group):

    def initialize(self):
        self.options.declare('E')
        self.options.declare('L')
        self.options.declare('b')
        self.options.declare('volume')
        self.options.declare('num_elements', int)

    def setup(self):
        E = self.options['E']
        L = self.options['L']
        b = self.options['b']
        volume = self.options['volume']
        num_elements = self.options['num_elements']
        num_nodes = num_elements + 1

        force_vector = np.zeros(2 * num_nodes)
        force_vector[-2] = -1.

        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('h', shape=num_elements)
        self.add_subsystem('inputs_comp', inputs_comp)

        I_comp = MomentOfInertiaComp(num_elements=num_elements, b=b)
        self.add_subsystem('I_comp', I_comp)

        # TODO: Add the rest of the components, following the XDSM
        comp = LocalStiffnessMatrixComp(num_elements=num_elements, E=E, L=L)
        self.add_subsystem('local_stiffness_matrix_comp', comp)

        comp = FEM(num_elements=num_elements,
                  force_vector=force_vector)
        self.add_subsystem('FEM', comp)

        comp = ComplianceComp(num_elements=num_elements, force_vector=force_vector)
        self.add_subsystem('compliance_comp', comp)

        comp = VolumeComp(num_elements=num_elements, b=b, L=L)
        self.add_subsystem('volume_comp', comp)

        ############################################
        # Connections between components
        ############################################
        self.connect('inputs_comp.h', 'I_comp.h')
        # TODO: connect I_comp -> local_stiffness
        #               local_stiffness -> FEM
        #               inputs_comp -> volume
        self.connect('I_comp.I', 'local_stiffness_matrix_comp.I')
        self.connect(
            'local_stiffness_matrix_comp.K_local',
            'FEM.K_local')

        self.connect(
            'inputs_comp.h',
            'volume_comp.h')

        # connection for: FEM -> compliance
        # this one is tricky, because you just want the states from the nodes,
        # but not the last 2 which relate to the clamped boundary condition on the left

        self.connect(
            'FEM.u',
            'compliance_comp.displacements', src_indices=np.arange(2*num_nodes))

        self.add_design_var('inputs_comp.h', lower=1e-2, upper=10.)
        self.add_objective('compliance_comp.compliance')
        self.add_constraint('volume_comp.volume', equals=volume)


if __name__ == "__main__":

    import time

    E = 1.
    L = 1.
    b = 0.1
    volume = 0.01

    num_elements = 5

    prob = om.Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['disp'] = True

    ######################################################
    # Use top level FD or CS to approximate derivatives
    ######################################################
    # prob.model.approx_totals(method='fd', step_calc="rel", step=1e-3)
    # prob.model.approx_totals(method='cs')

    prob.setup()

    start_time = time.time()
    prob.run_driver()
    print('opt time', time.time()-start_time)

    print(prob['inputs_comp.h'])
