from __future__ import print_function, division, absolute_import

import numpy as np
import openmdao.api as om
from openmdao.test_suite.test_examples.beam_optimization.beam_group import BeamGroup


E = 1.
L = 1.
b = 0.1
volume = 0.01

num_elements = 15

prob = om.Problem(model=BeamGroup(E=E, L=L, b=b, volume=volume, num_elements=num_elements))

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9
prob.driver.options['disp'] = True

prob.setup(force_alloc_complex=False)

prob['inputs_comp.h'][:] = 1.0


prob.run_driver()

# prob.model.list_outputs(print_arrays=True)

print('Optimal element height distribution:')
print(prob['inputs_comp.h'])
