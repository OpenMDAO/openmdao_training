from __future__ import division
import openmdao.api as om
# the __future__ import forces float division by default in Python2
# the openmdao.api import loads baseclasses from OpenMDAO

class BatteryWeight(om.ExplicitComponent):
    """Computes battery weight explicitly from desired range"""

    def initialize(self):
        self.options.declare('g', default=9.81) # options do not have units - careful!

    def setup(self):
        # Inputs
        self.add_input('LoverD', 20.0, units=None, desc="Lift to drag ratio")
        self.add_input('TOW', 6000, units='lbm', desc="Battery weight")
        self.add_input('eta_electric', 0.92, units=None, desc="Electric propulsion system efficiency")
        self.add_input('eta_prop', 0.8, units=None, desc="Propulsive efficiency")
        self.add_input('spec_energy', 300, units='W * h / kg', desc="Battery specific energy")
        self.add_input('range_desired', 150, units="NM", desc="Breguet range") # case sensitive - nm = nanometers

        # Outputs
        self.add_output('W_battery', 1500, units='lbm', desc="Takeoff weight")

        # TODO define finite difference derivatives for this component by adding method='fd' to the following declaration
        self.declare_partials('W_battery',['*'], method='fd')

    def compute(self, inputs, outputs):
        g = self.options['g']
        outputs['W_battery'] = inputs['TOW'] / (inputs['LoverD'] *
                                                      inputs['eta_electric'] * inputs['eta_prop'] *
                                                      inputs['spec_energy'] / g / inputs['range_desired'])

class WeightBuild(om.ExplicitComponent):
    """Compute TOW from component weights"""

    def setup(self):
        # define the following inputs: W_payload, W_empty, TOW
        self.add_input('W_payload', 800, units='lbm')
        self.add_input('W_empty', 5800, units='lbm')
        self.add_input('W_battery', 1500, units='lbm')

        # define the following outputs: W_battery
        self.add_output('TOW', val=6000, units='lbm')

        # declare generic finite difference partials
        self.declare_partials('TOW',['*'],method='fd')

    def compute(self, inputs, outputs):
        # implement the calculation W_battery = TOW - W_payload - W_empty
        outputs['TOW'] = inputs['W_battery'] + inputs['W_payload'] + inputs['W_empty']

    def compute_partials(self, inputs, partials):
        # TODO define partial derivatives here
        partials['TOW','W_battery'] = 1
        partials['TOW','W_payload'] = 1
        partials['TOW','W_empty'] = 1


class WeightBuildImplicit(om.ImplicitComponent):
    """Compute TOW from component weights"""

    def setup(self):
        # define the following inputs: W_payload, W_empty, TOW
        self.add_input('W_payload', 800, units='lbm')
        self.add_input('W_empty', 5800, units='lbm')
        self.add_input('W_battery', 1500, units='lbm')

        # define the following outputs: W_battery
        self.add_output('TOW', val=6000, units='lbm')

        # declare generic finite difference partials
        self.declare_partials('TOW',['*'],method='fd')

    def apply_nonlinear(self, inputs, outputs, residuals):
        # implement the calculation W_battery = TOW - W_payload - W_empty
        residuals['TOW'] = (inputs['W_battery'] + inputs['W_payload'] + inputs['W_empty']) - outputs['TOW']

    def linearize(self, inputs, outputs, partials):
        # TODO define partial derivatives here
        partials['TOW','W_battery'] = 1
        partials['TOW','W_payload'] = 1
        partials['TOW','W_empty'] = 1
        partials['TOW','TOW'] = -1


class ElecRangeGroup(om.Group):
    """A model to compute the max range of an electric aircraft
       Uses only ExplicitComponents"""

    def setup(self):
        # set some input values - optimizers act on independent variables
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes_outputs=['*'])
        indeps.add_output('W_payload', 800, units="lbm")
        indeps.add_output('range_desired', 150, units="NM")
        indeps.add_output('LoverD', 20)
        indeps.add_output('eta_electric', 0.92)
        indeps.add_output('eta_prop', 0.83)
        indeps.add_output('spec_energy', 300, units='W * h / kg')

        # add your disciplinary models to the group

        # The ExecComp lets you define an ad-hoc component without having to make a class
        self.add_subsystem('oew', om.ExecComp('W_empty=0.6*TOW',
                                              W_empty={'value':3500, 'units':'lbm'},
                                              TOW={'value':6000,'units':'lbm'}),
                                              promotes_outputs=['W_empty'])

        self.add_subsystem('batterywt', BatteryWeight(),
                           promotes_inputs=['LoverD','eta*','spec_energy'],
                           promotes_outputs=['*'])
        self.connect('range_desired','batterywt.range_desired')
        # self.add_subsystem('tow' ,WeightBuild(), promotes_inputs=['W_*'])
        self.add_subsystem('tow' ,WeightBuildImplicit(), promotes_inputs=['W_*'])
        self.connect('tow.TOW', ['batterywt.TOW','oew.TOW'])



if __name__ == "__main__":
    prob = om.Problem()

    prob.model = ElecRangeGroup()

    # pick a solver: 'newton', 'broyden', 'nlbgs', or 'nlbjac'
    # must define a nonlinear solver since this system has a cycle

    solver_flag = 'newton'

    if solver_flag == 'newton':
        prob.model.nonlinear_solver=om.NewtonSolver(iprint=2)
        # solve_subsystems should almost always be turned on
        # it improves solver robustness
        prob.model.nonlinear_solver.options['solve_subsystems'] = True
        prob.model.nonlinear_solver.options['maxiter'] = 100
        # these options control how tightly the solver converges the system
        prob.model.nonlinear_solver.options['atol'] = 1e-8
        prob.model.nonlinear_solver.options['rtol'] = 1e-8
        # the Newton solver requires a linear solver
        prob.model.linear_solver = om.DirectSolver()

    elif solver_flag == 'broyden':
        prob.model.nonlinear_solver=om.BroydenSolver(iprint=2)
        # TODO: Try using broyden with and without a computed jacobian. What happens?
        prob.model.nonlinear_solver.options['compute_jacobian'] = True
        prob.model.nonlinear_solver.options['maxiter'] = 100
        # these options control how tightly the solver converges the system
        prob.model.nonlinear_solver.options['atol'] = 1e-8
        prob.model.nonlinear_solver.options['rtol'] = 1e-8
        # the Broyden solver requires a linear solver *if* options['compute_jacobian'] = True
        prob.model.linear_solver = om.DirectSolver()

    elif solver_flag == 'nlbgs':
        # The nonlinear block Gauss-Seidel solver is an iterative solvver
        # Requires no linear solver and works even without derivatives
        prob.model.nonlinear_solver=om.NonlinearBlockGS(iprint=2)
        prob.model.nonlinear_solver.options['maxiter'] = 400
        prob.model.nonlinear_solver.options['atol'] = 1e-8
        prob.model.nonlinear_solver.options['rtol'] = 1e-8
        # The Aitken relaxation method improves robustness at cost of some speed
        prob.model.nonlinear_solver.options['use_aitken'] = False
        prob.model.nonlinear_solver.options['use_apply_nonlinear'] = True

    elif solver_flag == 'nlbjac':
        # We don't usually recommend using nonlinear block Jacobi as it converges slower
        prob.model.nonlinear_solver=om.NonlinearBlockJac(iprint=2)
        prob.model.nonlinear_solver.options['maxiter'] = 400
        prob.model.nonlinear_solver.options['atol'] = 1e-8
        prob.model.nonlinear_solver.options['rtol'] = 1e-8

    else:
        raise ValueError("bad solver selection!")

    prob.setup()
    ### If using the Newton solver you should generally check your partial derivatives
    ### before you run the model. It won't converge if you made a mistake.
    # prob.check_partials(compact_print=True)

    prob.run_model()

    ### If you want to list all inputs and outputs, uncomment the following
    # prob.model.list_inputs(units=True)
    # prob.model.list_outputs(units=True, residuals=True)
    print('Takeoff weight: ')
    print(str(prob['tow.TOW']) + ' lbs')
