from __future__ import division
import openmdao.api as om

# the __future__ import forces float division by default in Python2
# the openmdao.api import loads baseclasses from OpenMDAO


class BatteryWeight(om.ExplicitComponent):
    """Compute the weight left over for batteries at TOW.
       You will need to complete this component"""

    def setup(self):
        # TODO: define the following inputs:
        # W_payload 800 lbm
        # W_empty 5800 lbm
        # TOW 12000 lbm
        # replace the <bracketed> placeholders including the brackets
        # do this for each input (3 calls)
        self.add_input(<name>, <default value>, units=<unit string>)

        # TODO: define the following outputs:
        # W_battery, 1000 lbm
        #
        self.add_output(<name>, <default value>, units=<unit string>)

        # TODO: declare generic finite difference partials (wildcards)
        self.declare_partials(<of name>, <wrt name>, method='fd')

    def compute(self, inputs, outputs):
        # TODO: implement the calculation W_battery = TOW - W_payload - W_empty
        outputs[''] = inputs[''] .....

class BreguetRange(om.ExplicitComponent):
    """Compute the Breguet range for an electric aircraft.
       This example component is pre-filled with the correct code."""

    def initialize(self):
        self.options.declare('g', default=9.81) # options do not have units - careful!

    def setup(self):
        # Inputs
        self.add_input('LoverD', 15.0, units=None, desc="Lift to drag ratio")
        self.add_input('W_battery', 1000, units='lbm', desc="Battery weight")
        self.add_input('TOW', 12000, units='lbm', desc="Takeoff weight")
        self.add_input('eta_electric', 0.92, units=None, desc="Electric propulsion system efficiency")
        self.add_input('eta_prop', 0.8, units=None, desc="Propulsive efficiency")
        self.add_input('spec_energy', 300, units='W * h / kg', desc="Battery specific energy")

        # Outputs
        self.add_output('range', 100, units="NM", desc="Breguet range") # case sensitive - nm = nanometers

        # Partial derivatives
        self.declare_partials('range', ['*'], method='fd')

    def compute(self, inputs, outputs):
        g = self.options['g']
        # outputs['range'] = # implement computation here
        outputs['range'] = (inputs['LoverD'] *
                            inputs['eta_electric'] * inputs['eta_prop'] *
                            inputs['spec_energy'] / g *
                            inputs['W_battery'] / inputs['TOW'])



class ElecRangeGroup(om.Group):
    """A model to compute the max range of an electric aircraft
       You will need to make connections between the components"""

    def setup(self):
        # set some input values - optimizers act on independent variables
        indeps = self.add_subsystem('indeps', om.IndepVarComp(), promotes_outputs=['*'])
        indeps.add_output('W_payload', 800, units="lbm")
        indeps.add_output('W_empty', 5800, units="lbm")
        indeps.add_output('MTOW', 12000, units="lbm")
        indeps.add_output('LoverD', 20)
        indeps.add_output('eta_electric', 0.92)
        indeps.add_output('eta_prop', 0.83)
        indeps.add_output('spec_energy', 300, units='W * h / kg')

        # add your disciplinary models to the group

        self.add_subsystem('batterywt', BatteryWeight())
        self.add_subsystem('breguet', BreguetRange())
        # TODO: finish these connections yourself with self.connect() or promotions
        # self.connect(<output_var>, <input_var>)
        # self.add_subsystems(..., promotes_inputs=['*'], promotes_outputs=['*'])
        # for this exercise we assume TOW = MTOW (max range at current payload)



if __name__ == "__main__":
    prob = om.Problem()
    prob.model = ElecRangeGroup()
    prob.setup()
    prob.run_model()
    print('Computed max range: ')
    print(str(prob['breguet.range']) + ' nautical miles')
