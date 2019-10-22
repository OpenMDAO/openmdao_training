from __future__ import division
import numpy as np

import openmdao.api as om

# all of these components have already been created for you,
# but look in beam_comp.py if you're curious to see how
from beam_comps import (MomentOfInertiaComp, LocalStiffnessMatrixComp, FEM,
                        ComplianceComp, VolumeComp)


class BeamGroup(om.Group):

    def initialize(self):
        self.options.declare('E', desc="Young's modulus")
        self.options.declare('L', desc="beam overall length")
        self.options.declare('b', desc='beam thickness')
        self.options.declare('num_elements', types=int, # this will force some type checking for you
                             desc="number of segments to break beam into")

        # TODO: Declare volume as an option to fix the error msg when instantiating this component

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
        # self.add_subsystem(...)

        ############################################
        # Connections between components
        ############################################
        self.connect('inputs_comp.h', 'I_comp.h')

        # TODO: connect I_comp -> local_stiffness
        #               local_stiffness -> FEM
        #               inputs_comp -> volume

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

    num_elements = 50

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
