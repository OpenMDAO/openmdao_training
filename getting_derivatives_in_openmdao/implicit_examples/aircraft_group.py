from __future__ import print_function
import numpy as np
import openmdao.api as om

# Import components from their files
from balanced_eom import BalancedEOM
from simple_wing import SimpleWing


# Set the number of analysis points
nn = 2

# Instantiate an OpenMDAO problem with an empty group as the model
prob = om.Problem(model=om.Group())

# Add an IndepVarComp to the model and promotes all variables
design_parameters = prob.model.add_subsystem('design_parameters',
                                             om.IndepVarComp(),
                                             promotes=['*'])

# Provide flight conditions, in this case for two analysis points
design_parameters.add_output('mass', [250.e3, 200.e3], units='kg')
design_parameters.add_output('velocity', [200., 250.], units='m/s')
design_parameters.add_output('gamma', [0., 0.], units='rad')
design_parameters.add_output('S_ref', 383.7 * np.ones(nn), units='m**s')
design_parameters.add_output('rho', [1.2, 0.4], units='kg/m**3')

# Add the drag polar and equations of motion components
prob.model.add_subsystem('simple_wing', SimpleWing(num_nodes=nn), promotes=['*'])
prob.model.add_subsystem('EOM', BalancedEOM(num_nodes=nn), promotes=['*'])

# Set the nonlinear and linear solvers on the top level; print solver convergence
prob.model.nonlinear_solver = om.NewtonSolver()
prob.model.linear_solver = om.DirectSolver()
prob.set_solver_print(level=2)

# Setup and run the model
prob.setup()
prob.run_model()

# List the outputs from the converged model
prob.model.list_outputs(print_arrays=True, units=True)
