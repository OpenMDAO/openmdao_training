from __future__ import print_function
import numpy as np
import openmdao.api as om

from balanced_eom import BalancedEOM
from simple_wing import SimpleWing


nn = 1

prob = om.Problem(model=om.Group())

design_parameters = prob.model.add_subsystem('design_parameters', om.IndepVarComp(), promotes=['*'])

design_parameters.add_output('mass', 250.e3 * np.ones(nn), units='kg')
design_parameters.add_output('velocity', 250., units='m/s')
design_parameters.add_output('gamma', 0., units='rad')
design_parameters.add_output('S_ref', 594720 * np.ones(nn), units='inch**2')
design_parameters.add_output('rho', 1.2 * np.ones(nn), units='kg/m**3')

prob.model.add_subsystem('simple_wing', SimpleWing(num_nodes=nn), promotes=['*'])
prob.model.add_subsystem('EOM', BalancedEOM(num_nodes=nn), promotes=['*'])


prob.model.nonlinear_solver = newton = om.NewtonSolver()
newton.linesearch = om.BoundsEnforceLS()

prob.model.linear_solver = om.DirectSolver()

prob.set_solver_print(level=2)

prob.setup(force_alloc_complex=True)

prob.run_model()

# prob.check_partials(method='cs', step=1e-40, compact_print=True)

prob.model.list_outputs(units=True)
